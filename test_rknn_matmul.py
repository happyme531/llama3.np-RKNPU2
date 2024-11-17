import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
from typing import Tuple, List
from rknn_matmul import RKNNMatMul, RKNNMatMulLayout, RKNNTensorType


def benchmark_matmul(sizes: List[Tuple[int, int, int]], 
                    num_runs: int = 10,
                    out_type: str = 'float32') -> None:
    """对比RKNN MatMul和NumPy的性能和精度
    
    Args:
        sizes: 要测试的矩阵大小列表,每个元素为(M, K, N)
        num_runs: 每个大小运行的次数
        out_type: 输出类型,'float32'或'float16'
    """
    rknn_mm = RKNNMatMul(perf_debug=False)
    out_tensor_type = RKNNTensorType.RKNN_TENSOR_FLOAT32 if out_type == 'float32' else RKNNTensorType.RKNN_TENSOR_FLOAT16
    
    print(f"\nBenchmarking RKNN MatMul vs NumPy (output type: {out_type})")
    print("-" * 80)
    print(f"{'Size':>15} {'RKNN Time':>12} {'NumPy Time':>12} {'Speedup':>10} {'Max Error':>12} {'Mean Error':>12}")
    print("-" * 80)
    
    for M, K, N in sizes:
        # 创建随机测试数据
        # a = np.random.randn(1, M, K).astype(np.float16)
        # b = np.random.randn(K, N).astype(np.float16)
        a = np.load("x.npy").squeeze().astype(np.float16)
        b = np.load("up_weight.npy").astype(np.float16)
        
        # RKNN MatMul计时
        rknn_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            c_rknn = rknn_mm.matmul(a, b, out_tensor_type, b_native_layout=True, const_b=True)
            rknn_times.append(time.perf_counter() - start)
        rknn_time = np.mean(rknn_times[1:]) * 1000  # 去掉第一次运行,转换为ms
        # a = a.astype(np.float32)
        # b = b.astype(np.float32)
        # NumPy计时
        numpy_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            c_numpy = np.matmul(a, b)
            if out_type == 'float16':
                c_numpy = c_numpy.astype(np.float16)
            numpy_times.append(time.perf_counter() - start)
        numpy_time = np.mean(numpy_times[1:]) * 1000  # 去掉第一次运行,转换为ms
        
        # 计算误差
        abs_diff = np.abs(c_rknn - c_numpy)
        max_error = np.max(abs_diff)
        mean_error = np.mean(abs_diff)
        
        # 打印结果
        size_str = f"{M}x{K}x{N}"
        speedup = numpy_time / rknn_time
        print(f"{size_str:>15} {rknn_time:>10.2f}ms {numpy_time:>10.2f}ms {speedup:>10.2f}x {max_error:>10.2e} {mean_error:>10.2e}")

def main():
    # 测试不同大小的矩阵
    sizes = [
        # (16, 32, 16),       
        # (128, 128, 128),    # 小矩阵
        # (512, 512, 512),    # 中等矩阵
        # (1024, 1024, 1024), # 大矩阵
        # # (2048, 2048, 2048), # 更大矩阵
        # # (4096, 4096, 4096), # 超大矩阵
        # (32, 4096, 1024), # 非方阵
        # (256, 256, 4096), # 非方阵
        # (16, 2048, 8192),
        (1, 2048, 8192)

    ]
    
    # 测试FP32输出
    benchmark_matmul(sizes, out_type='float32')
    
    # 测试FP16输出
    benchmark_matmul(sizes, out_type='float16')

if __name__ == '__main__':
    main()