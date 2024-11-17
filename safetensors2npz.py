import argparse
import torch
from safetensors import safe_open
import numpy as np

def convert_safetensors_to_npz(input_path, output_path):
    # 使用safetensors加载模型到PyTorch
    tensors = {}
    with safe_open(input_path, framework="pt") as f:
        for key in f.keys():
            # 加载到PyTorch tensor
            tensor = f.get_tensor(key)
            # 转换为float32，然后转为numpy，最后转为float16
            np_tensor = tensor.float().numpy().astype(np.float32)
            tensors[key] = np_tensor
            print(f"Converting: {key}, shape: {np_tensor.shape}, dtype: {np_tensor.dtype}")
    
    # 保存为压缩的npz格式
    np.savez(output_path, **tensors)
    print(f"\nSuccessfully converted {input_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert SafeTensors to NPZ format with FP16 precision')
    parser.add_argument('input', help='Input safetensors file path')
    parser.add_argument('output', help='Output npz file path')
    
    args = parser.parse_args()
    convert_safetensors_to_npz(args.input, args.output)

if __name__ == "__main__":
    main()