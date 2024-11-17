from __future__ import annotations

import math
import os
import sys
import time
from typing import TypeVar, Generic, Optional
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters

from rknn_matmul import RKNNMatMul, RKNNTensorType

Shape = TypeVar("Shape")


class Array(np.ndarray, Generic[Shape]): ...

import numpy as np
import os

def visualize_array(arr, save_path, normalize=True, cmap=cv2.COLORMAP_TWILIGHT):
    """
    将形状为(1,w,h)或(w,h)的numpy数组保存为图像文件
    
    参数:
        arr: numpy数组,形状为(1,w,h)或(w,h)
        save_path: 图像保存路径,支持.png/.jpg等格式
        normalize: 是否将数据归一化到[0,1]范围
        cmap: OpenCV颜色映射方案,默认为COLORMAP_VIRIDIS
        dpi: 图像分辨率,默认300
    """
    # 确保输入是numpy数组
    arr = np.asarray(arr)
    
    # 处理维度
    arr = np.squeeze(arr)  # 移除大小为1的维度
    
    # 归一化到0-255范围
    if normalize:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr_norm = (arr * 255).astype(np.uint8)
    
    # 应用颜色映射
    colored = cv2.applyColorMap(arr_norm, cmap)
    
    # 创建colorbar
    colorbar_width = 30
    colorbar = np.linspace(0, 255, colored.shape[0]).astype(np.uint8)
    colorbar = np.tile(colorbar[:, np.newaxis], (1, colorbar_width))
    colorbar = cv2.applyColorMap(colorbar, cmap)
    
    # 添加值标签
    if normalize:
        min_val, max_val = 0, 1
    else:
        min_val, max_val = arr.min(), arr.max()
    
    # 合并图像和colorbar
    padding = np.zeros((colored.shape[0], 20, 3), dtype=np.uint8)  # 添加间距
    result = np.hstack([colored, padding, colorbar])
    
    # 添加文本标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    cv2.putText(result, f'{max_val:.2f}', 
                (colored.shape[1] + padding.shape[1] - 10, 20), 
                font, font_scale, (255,255,255), 1)
    cv2.putText(result, f'{min_val:.2f}', 
                (colored.shape[1] + padding.shape[1] - 10, result.shape[0] - 10), 
                font, font_scale, (255,255,255), 1)
    
    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 保存图像
    cv2.imwrite(save_path, result)

def compute_error(a, b):
    abs_diff = np.abs(a - b)
    max_error = np.max(abs_diff)
    mean_error = np.mean(abs_diff)
    print(f"max_error: {max_error}, mean_error: {mean_error}")

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)).astype(np.float32)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def rotate_half(x: Array["..., D"]):
    """Rotates half the hidden dims of the input."""
    d = x.shape[-1]
    x1, x2 = np.split(x, 2, axis=-1) 
    return np.concatenate((-x2, x1), axis=-1)


def compute_cos_sin_cache(head_dim: int, max_seq_len: int, base: int = 500000):
    """
    Compute cos and sin cache for rotary embeddings.
    Aligned with PyTorch version's default RoPE implementation.
    """
    # Compute inverse frequency bands
    inv_freq = (1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)))
    
    # Compute position frequencies 
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)  # [seq_len, head_dim//2]
    
    # Compute cos and sin and duplicate to match full head_dim
    cos = np.cos(freqs)  # [seq_len, head_dim//2]
    sin = np.sin(freqs)  # [seq_len, head_dim//2]
    cos = np.concatenate([cos, cos], axis=1)  # [seq_len, head_dim]
    sin = np.concatenate([sin, sin], axis=1)  # [seq_len, head_dim]
    
    return cos.astype(np.float32), sin.astype(np.float32)


def apply_rotary_emb(xq: Array["B, L or 1, QHN, HD"], xk: Array["B, L or 1, KVHN, HD"],
                     freqs_cos: Array["L or 1, HD"], freqs_sin: Array["L or 1, HD"]):
    """
    Apply rotary embeddings to query and key tensors.
    Aligned with PyTorch version's apply_rotary_pos_emb function.
    """
    # Add missing broadcast dimensions to match PyTorch version
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))  # [1, seq_len, 1, head_dim]
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))  # [1, seq_len, 1, head_dim]
    
    # Apply rotation using the helper function
    xq_out = (xq * freqs_cos) + (rotate_half(xq) * freqs_sin)
    xk_out = (xk * freqs_cos) + (rotate_half(xk) * freqs_sin)
    
    return xq_out, xk_out


def repeat_kv(x: Array["B, L, KVHN, HD"], n_rep: int):
    if n_rep == 1:
        return x
    z: Array["B, L, QHN, HD"] = np.repeat(x, n_rep, axis=2)
    return z


class FeedForward:
    def __init__(self, up_weight: Array["FD, D"], gate_weight: Array["FD, D"], down_weight: Array["D, FD"]):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T
        self.mm_gate = RKNNMatMul()
        self.mm_up = RKNNMatMul()
        self.mm_down = RKNNMatMul()

    def __call__(self, x: Array["B, L or 1, D"]):
        # FD = 2 * 4 * D / 3
        # swish: Array["B, L or 1, FD"] = silu(x @ self.gate_weight)
        # x_V: Array["B, L or 1, FD"] = x @ self.up_weight
        # x: Array["B, L or 1, FD"] = swish * x_V
        # x: Array["B, L or 1, D"] = x @ self.down_weight
        gate = self.mm_gate.matmul(x, self.gate_weight, RKNNTensorType.RKNN_TENSOR_FLOAT32, const_b=True, b_native_layout=True)
        swish = silu(gate)
        x_V = self.mm_up.matmul(x, self.up_weight, RKNNTensorType.RKNN_TENSOR_FLOAT32, const_b=True, b_native_layout=True)
        x = swish * x_V
        x = self.mm_down.matmul(x, self.down_weight, RKNNTensorType.RKNN_TENSOR_FLOAT32, const_b=True, b_native_layout=True)
        return x


class RMSNorm:
    def __init__(self, weight: Array["H"], eps: float):
        self.weight = weight.astype(np.float32)
        self.eps = np.float32(eps)

    def __call__(self, x: Array["B, L or 1, D"]):
        z = (x ** 2).mean(-1, keepdims=True).astype(np.float32) + self.eps
        z = x / np.sqrt(z)
        return (z * self.weight).astype(np.float32)


class Attention:
    def __init__(self, q_weight: Array["D, D"], k_weight: Array["D, D"], v_weight: Array["D, D"],
                 o_weight: Array["D, D"], args: ModelArgs):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T

        self.mm_q = RKNNMatMul()
        self.mm_k = RKNNMatMul()
        self.mm_v = RKNNMatMul()
        self.mm_o = RKNNMatMul()

        self.cache_k = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), dtype=np.float32)
        self.cache_v = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), dtype=np.float32)

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos: Array["L or 1, HD"], freqs_sin: Array["L or 1, HD"]):
        B, L, _ = x.shape

        # QKV
        # xq: Array["B, L or 1, D"] = x @ self.q_weight
        # xk: Array["B, L or 1, D"] = x @ self.k_weight
        # xv: Array["B, L or 1, D"] = x @ self.v_weight
        xq = self.mm_q.matmul(x, self.q_weight, RKNNTensorType.RKNN_TENSOR_FLOAT32, const_b=True, b_native_layout=True)
        xk = self.mm_k.matmul(x, self.k_weight, RKNNTensorType.RKNN_TENSOR_FLOAT32, const_b=True, b_native_layout=True)
        xv = self.mm_v.matmul(x, self.v_weight, RKNNTensorType.RKNN_TENSOR_FLOAT32, const_b=True, b_native_layout=True)

        xq: Array["B, L or 1, QHN,  HD"] = xq.reshape(B, L, self.n_local_heads, self.head_dim)
        xk: Array["B, L or 1, KVHN, HD"] = xk.reshape(B, L, self.n_local_kv_heads, self.head_dim)
        xv: Array["B, L or 1, KVHN, HD"] = xv.reshape(B, L, self.n_local_kv_heads, self.head_dim)

        # RoPE #2
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # KV Cache
        self.cache_k[:B, start_pos: start_pos + L] = xk
        self.cache_v[:B, start_pos: start_pos + L] = xv
        ks: Array["B, L, KVHN, HD"] = self.cache_k[:B, : start_pos + L]
        vs: Array["B, L, KVHN, HD"] = self.cache_v[:B, : start_pos + L]

        # GQA
        xk: Array["B, L, HN, HD"] = repeat_kv(ks, self.n_rep)
        xv: Array["B, L, HN, HD"] = repeat_kv(vs, self.n_rep)

        # ["B, L, HN, HD"] -> ["B, HN, L, HD"]
        xq: Array["B, HN, L or 1, HD"] = xq.transpose(0, 2, 1, 3)
        xk: Array["B, HN, L, HD"] = xk.transpose(0, 2, 1, 3)
        xv: Array["B, HN, L, HD"] = xv.transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        # ["B, HN, L or 1, HD"] @ ["B, HN, HD, L"] -> ["B, HN, L or 1, L"]
        attention: Array["B, HN, L or 1, L"] = xq @ xk.transpose(0, 1, 3, 2) / np.float32(math.sqrt(self.head_dim))
        # np.savez("qka.npz", query_states=xq, key_states=xk, attn_weights=attention)
        # `mask` is used only once at the beginning.
        if mask is not None:
            attention = attention + mask[None, None, :, :]
        attention = softmax(attention)
        output: Array["B, HN, L or 1, HD"] = attention @ xv

        # ["B, HN, L or 1, HD"] -> ["B, L or 1, D"]
        output: Array["B, L or 1, D"] = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        # output: Array["B, L or 1, D"] = output @ self.o_weight
        output = self.mm_o.matmul(output, self.o_weight, RKNNTensorType.RKNN_TENSOR_FLOAT32, const_b=True, b_native_layout=True)

        return output


class TransformerBlock:
    def __init__(self, weight: dict, layer_id: int, args: ModelArgs):
        self.attention = Attention(
            weight.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            args
        )
        self.feed_forward = FeedForward(
            weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
        )
        self.input_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=args.norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
            eps=args.norm_eps
        )

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Array["L, L"],
                 freqs_cos: Array["L or 1, HD"], freqs_sin: Array["L or 1, HD"]):
        # RMSNorm
        norm_x: Array["B, L or 1, D"] = self.input_layernorm(x)
        # Masked Multi-Head Attention
        h1: Array["B, L or 1, D"] = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = x + h1

        # RMSNorm
        norm_z = self.post_attention_layernorm(z)
        # Feed Forward + SwiGLU
        h2: Array["B, L or 1, D"] = self.feed_forward(norm_z)
        out = z + h2

        return out


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args

        weight = load_parameters(model_path)
        self.tok_embedding: Array["VS, D"] = weight.get("model.embed_tokens.weight")

        # RoPE #1
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(args.dim // args.n_heads, args.max_seq_len)

        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(weight, layer_id, args))

        self.norm = RMSNorm(weight.get("model.norm.weight"), eps=args.norm_eps)
        # self.lm_head_weight: Array["D, VS"] = weight.get("lm_head.weight").T
        self.lm_head_weight: Array["D, VS"] = self.tok_embedding.T

        self.mm_lm_head = RKNNMatMul(iommu_domain_id=1)

        del weight

    def __call__(self, input_ids: Array["B, L"], start_pos: int):
        _, L = input_ids.shape
        h: Array["B, L or 1, D"] = self.tok_embedding[input_ids]
        # ["M, HD//2"] -> ["L or 1, HD//2"]
        freqs_cos: Array["L or 1, HD"] = self.freqs_cos[start_pos: start_pos + L]
        freqs_sin: Array["L or 1, HD"] = self.freqs_sin[start_pos: start_pos + L]

        # `mask` is generated only once at the beginning.
        mask: Array["L, L"] = None
        if L > 1:
            mask = np.full((L, L), np.float32('-inf'))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((L, start_pos), dtype=np.float32), mask], axis=1)

        # Transformer Layers
        for i, layer in enumerate(self.layers):
            h: Array["B, L or 1, D"] = layer(h, start_pos, mask, freqs_cos, freqs_sin)

        # RMSNorm
        h: Array["B, L or 1, D"] = self.norm(h)
        # Only forward the output from the last position.
        # ["B, 1, VS"] = ["B, 1(L), D"] @ ["D, VS"]
        # logit: Array["B, 1, VS"] = h[:, [-1], :] @ self.lm_head_weight
        logit = self.mm_lm_head.matmul(h, self.lm_head_weight, RKNNTensorType.RKNN_TENSOR_FLOAT32, const_b=True, b_native_layout=True)
        return logit

    def generate(self, input_ids: Array["B, L"], max_new_tokens: int):
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, L + max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id.reshape(1, -1)
                pos = curr_pos
            logits: Array["B, 1, VS"] = self(inputs, pos)
            
            # 添加温度采样
            logits = logits / 0.7
            probs = softmax(logits)
            # 生成正确形状的 next_id
            next_id = np.random.choice(
                probs.shape[-1], 
                size=1,
                p=probs[0, -1]
            ).reshape(1, 1)
            
            # 检查是否生成了任何结束标记
            if next_id.item() in [128001, 128008, 128009]:  # 使用 item() 安全地获取标量值
                break
            
            yield next_id


from transformers import AutoTokenizer
def main():
    args = ModelArgs()

    # tokenizer = Tokenizer("./tokenizer.model.np")
    tokenizer = AutoTokenizer.from_pretrained("../llama3.2")
    model = Llama("/mnt/ssd1/home/user/llama3.2-1b.npz", args)

    if len(sys.argv) == 1:
        prompt = "Here is a python script of a algorithm QuickSort algorithm: \n```python"
    else:
        prompt = sys.argv[1]

    print(f"\n{prompt}", end="")
    input_ids = np.array([tokenizer.encode(prompt)])
    # print(input_ids)
    start = None
    _, L = input_ids.shape
    for id in model.generate(input_ids, args.max_new_tokens):
        if L == input_ids.shape[1] + 1:
            start = time.time()
        L += 1
        output_id = id.item()
        if output_id in [128001, 128008, 128009]:
            break
        print(tokenizer.decode([output_id]), end="")
        sys.stdout.flush()
    elapsed = time.time() - start
    L -= input_ids.shape[1]
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s")

import cProfile
# cProfile.run('main()', sort='tottime')
main()
