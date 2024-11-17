import ctypes
import numpy as np
from enum import IntEnum
from typing import Optional, Tuple, Union
import time

# 加载RKNN Runtime库
_lib = ctypes.CDLL("librknnrt.so")

# 加载layout转换库
_layout_lib = ctypes.CDLL("./librknn_matmul_layout_convert.so")

RKNN_MAX_NAME_LEN = 256
RKNN_MAX_DIMS = 16

class RKNNTensorType(IntEnum):
    RKNN_TENSOR_FLOAT32 = 0
    RKNN_TENSOR_FLOAT16 = 1
    RKNN_TENSOR_INT8 = 2
    RKNN_TENSOR_UINT8 = 3
    RKNN_TENSOR_INT16 = 4
    RKNN_TENSOR_UINT16 = 5
    RKNN_TENSOR_INT32 = 6
    RKNN_TENSOR_UINT32 = 7
    RKNN_TENSOR_INT64 = 8

class RKNNMatMulType(IntEnum):
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1
    RKNN_INT8_MM_INT8_TO_INT32 = 2
    RKNN_INT8_MM_INT8_TO_INT8 = 3
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 = 4
    RKNN_FLOAT16_MM_INT8_TO_FLOAT32 = 5
    RKNN_FLOAT16_MM_INT8_TO_FLOAT16 = 6
    RKNN_FLOAT16_MM_INT4_TO_FLOAT32 = 7
    RKNN_FLOAT16_MM_INT4_TO_FLOAT16 = 8
    RKNN_INT8_MM_INT8_TO_FLOAT32 = 9
    RKNN_INT4_MM_INT4_TO_INT16 = 10
    RKNN_INT8_MM_INT4_TO_INT32 = 11

class RKNNMatMulLayout(IntEnum):
    RKNN_MM_LAYOUT_NORM = 0
    RKNN_MM_LAYOUT_NATIVE = 1 
    RKNN_MM_LAYOUT_TP_NORM = 2

class RKNNMatMulQuantType(IntEnum):
    RKNN_QUANT_TYPE_PER_LAYER_SYM = 0
    RKNN_QUANT_TYPE_PER_LAYER_ASYM = 1
    RKNN_QUANT_TYPE_PER_CHANNEL_SYM = 2
    RKNN_QUANT_TYPE_PER_CHANNEL_ASYM = 3
    RKNN_QUANT_TYPE_PER_GROUP_SYM = 4
    RKNN_QUANT_TYPE_PER_GROUP_ASYM = 5

# 从rknn_api.h
class RKNNTensorMem(ctypes.Structure):
    _fields_ = [
        ('virt_addr', ctypes.c_void_p),
        ('phys_addr', ctypes.c_uint64),
        ('fd', ctypes.c_int32),
        ('offset', ctypes.c_int32),
        ('size', ctypes.c_uint32),
        ('flags', ctypes.c_uint32),
        ('priv_data', ctypes.c_void_p)
    ]

class RKNNMatMulTensorAttr(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char * RKNN_MAX_NAME_LEN),
        ('n_dims', ctypes.c_uint32),
        ('dims', ctypes.c_uint32 * RKNN_MAX_DIMS),
        ('size', ctypes.c_uint32),
        ('type', ctypes.c_int32)
    ]

class RKNNMatMulIOAttr(ctypes.Structure):
    _fields_ = [
        ('A', RKNNMatMulTensorAttr),
        ('B', RKNNMatMulTensorAttr),
        ('C', RKNNMatMulTensorAttr)
    ]

class RKNNMatMulInfo(ctypes.Structure):
    _fields_ = [
        ('M', ctypes.c_int32),
        ('K', ctypes.c_int32),
        ('N', ctypes.c_int32),
        ('type', ctypes.c_int32),
        ('B_layout', ctypes.c_int16),
        ('B_quant_type', ctypes.c_int16),
        ('AC_layout', ctypes.c_int16),
        ('AC_quant_type', ctypes.c_int16),
        ('iommu_domain_id', ctypes.c_int32),
        ('group_size', ctypes.c_int16),
        ('reserved', ctypes.c_int8 * 34)
    ]

# 定义API函数参数和返回类型
_lib.rknn_matmul_create.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),  # ctx
    ctypes.POINTER(RKNNMatMulInfo),   # info  
    ctypes.POINTER(RKNNMatMulIOAttr)  # io_attr
]
_lib.rknn_matmul_create.restype = ctypes.c_int

_lib.rknn_matmul_destroy.argtypes = [ctypes.c_uint64]  # ctx
_lib.rknn_matmul_destroy.restype = ctypes.c_int

_lib.rknn_create_mem.argtypes = [
    ctypes.c_uint64,  # ctx
    ctypes.c_uint32   # size
]
_lib.rknn_create_mem.restype = ctypes.POINTER(RKNNTensorMem)

_lib.rknn_destroy_mem.argtypes = [
    ctypes.c_uint64,              # ctx
    ctypes.POINTER(RKNNTensorMem) # mem
]
_lib.rknn_destroy_mem.restype = ctypes.c_int

_lib.rknn_matmul_set_io_mem.argtypes = [
    ctypes.c_uint64,                    # ctx
    ctypes.POINTER(RKNNTensorMem),      # mem
    ctypes.POINTER(RKNNMatMulTensorAttr)# attr
]
_lib.rknn_matmul_set_io_mem.restype = ctypes.c_int

_lib.rknn_matmul_run.argtypes = [ctypes.c_uint64]  # ctx
_lib.rknn_matmul_run.restype = ctypes.c_int

_lib.rknn_B_normal_layout_to_native_layout.argtypes = [
    ctypes.c_void_p,      # B_input
    ctypes.c_void_p,      # B_output
    ctypes.c_int,         # K
    ctypes.c_int,         # N
    ctypes.POINTER(RKNNMatMulInfo)  # info
]
_lib.rknn_B_normal_layout_to_native_layout.restype = ctypes.c_int

# 定义函数原型
_layout_lib.fp16_norm_to_native_layout.argtypes = [
    ctypes.c_void_p,  # src
    ctypes.c_void_p,  # dst 
    ctypes.c_int32,   # K
    ctypes.c_int32    # N
]
_layout_lib.fp16_norm_to_native_layout.restype = ctypes.c_int

class RKNNTensorMemory:
    """管理RKNN tensor memory的类"""
    def __init__(self, ctx: ctypes.c_uint64, size: Optional[int] = None, np_array: Optional[np.ndarray] = None):
        """
        初始化RKNN tensor memory
        
        Args:
            ctx: RKNN context
            size: 内存大小(字节)
            np_array: NumPy数组,如果提供则从数组创建内存
        """
        self.ctx = ctx
        self._mem = None
        
        if np_array is not None:
            size = np_array.nbytes
            
        if size:
            # print(f"Allocating memory of size {size}")
            self._mem = _lib.rknn_create_mem(self.ctx, size)
            if not self._mem:
                raise RuntimeError(f"Failed to allocate memory of size {size}")
                
            if np_array is not None:
                self.update_from_numpy(np_array)
    
    def __del__(self):
        """析构时释放内存"""
        self.release()
    
    def release(self):
        """释放内存"""
        if self._mem:
            # print(f"Releasing memory of size {self._mem.contents.size}")
            _lib.rknn_destroy_mem(self.ctx, self._mem)
            self._mem = None
    
    @property
    def ptr(self) -> ctypes.POINTER(RKNNTensorMem):
        """获取底层内存指针"""
        return self._mem
    
    def update_from_numpy(self, array: np.ndarray) -> None:
        """从NumPy数组更新内存内容
        
        Args:
            array: 源NumPy数组
        """
        if not self._mem:
            raise RuntimeError("Memory not allocated")
            
        if array.nbytes > self._mem.contents.size:
            raise ValueError(f"Array size ({array.nbytes}) exceeds allocated memory ({self._mem.contents.size})")
            
        ctypes.memmove(self._mem.contents.virt_addr, array.ctypes.data, array.nbytes)
    
    def to_numpy(self, shape: Optional[Tuple[int, ...]] = None, dtype: np.dtype = np.float32) -> np.ndarray:
        """将内存内容转换为NumPy数组
        
        Args:
            shape: 数组形状,如果为None则创建一维数组
            dtype: 数组数据类型
        
        Returns:
            NumPy数组
        """
        if not self._mem:
            raise RuntimeError("Memory not allocated")
            
        buffer = ctypes.string_at(self._mem.contents.virt_addr, self._mem.contents.size)
        array = np.frombuffer(buffer, dtype=dtype)
        
        if shape is not None:
            array = array.reshape(shape)
            
        return array

class RKNNMatMul:
    def __init__(self, perf_debug: bool = False, iommu_domain_id: int = 0):
        self.ctx = ctypes.c_uint64(0)
        self._last_shape = None
        self._a_mem = None 
        self._b_mem = None
        self._c_mem = None
        self._io_attr = None
        self._out_type = None
        self._b_layout = RKNNMatMulLayout.RKNN_MM_LAYOUT_NORM
        self.perf_debug = perf_debug  # 添加性能调试开关
        self.iommu_domain_id = iommu_domain_id  # 存储iommu_domain_id
        
    def __del__(self):
        # RKNNTensorMemory会自动释放内存
        self._a_mem = None
        self._b_mem = None
        self._c_mem = None
        if self.ctx.value != 0:
            _lib.rknn_matmul_destroy(self.ctx)

    def _init_context(self, M: int, K: int, N: int, out_type: RKNNTensorType) -> None:
        """初始化matmul上下文和内存"""
        # 创建matmul信息
        info = RKNNMatMulInfo()
        info.M = M
        info.K = K 
        info.N = N
        info.type = RKNNMatMulType.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 if out_type == RKNNTensorType.RKNN_TENSOR_FLOAT32 else RKNNMatMulType.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16
        info.B_layout = self._b_layout
        info.B_quant_type = RKNNMatMulQuantType.RKNN_QUANT_TYPE_PER_LAYER_SYM
        info.AC_layout = RKNNMatMulLayout.RKNN_MM_LAYOUT_NORM
        info.AC_quant_type = RKNNMatMulQuantType.RKNN_QUANT_TYPE_PER_LAYER_SYM
        info.iommu_domain_id = self.iommu_domain_id  # 使用存储的iommu_domain_id
        info.group_size = 0

        # 创建IO属性
        self._io_attr = RKNNMatMulIOAttr()
        
        # 创建context
        if self.ctx.value != 0:
            self._a_mem = None
            self._b_mem = None
            self._c_mem = None  
            _lib.rknn_matmul_destroy(self.ctx)
            self.ctx = ctypes.c_uint64(0)
            
        ret = _lib.rknn_matmul_create(ctypes.byref(self.ctx), 
                                     ctypes.byref(info),
                                     ctypes.byref(self._io_attr))
        if ret < 0:
            raise RuntimeError(f"Failed to create matmul context: {ret}")
            
        # 分配内存
        self._a_mem = RKNNTensorMemory(self.ctx, self._io_attr.A.size)
        self._b_mem = RKNNTensorMemory(self.ctx, self._io_attr.B.size)
        self._c_mem = RKNNTensorMemory(self.ctx, self._io_attr.C.size)
        
        # 保存当前配置
        self._last_shape = (M, K, N)
        self._out_type = out_type

    def _convert_b_to_native_layout(self, b: np.ndarray) -> np.ndarray:
        """将矩阵B转换为native layout
        
        Args:
            b: 输入矩阵
            info: matmul信息
            
        Returns:
            转换后的矩阵
        """
        K, N = b.shape
        # 创建输出缓冲区
        b_native = np.empty_like(b)
        
        # 根据数据类型选择转换函数
        if b.dtype == np.float16:
            ret = _layout_lib.fp16_norm_to_native_layout(
                b.ctypes.data,
                b_native.ctypes.data,
                K,
                N
            )
        elif b.dtype == np.int8:
            ret = _layout_lib.int8_norm_to_native_layout(
                b.ctypes.data,
                b_native.ctypes.data, 
                K,
                N
            )
        else:
            raise ValueError(f"Unsupported data type: {b.dtype}")
        
        if ret < 0:
            raise RuntimeError(f"Failed to convert matrix B to native layout: {ret}")
            
        return b_native

    def matmul_2d(self, 
               a: np.ndarray, 
               b: np.ndarray,
               out_type: RKNNTensorType = RKNNTensorType.RKNN_TENSOR_FLOAT32,
               const_a: bool = False,
               const_b: bool = False,
               b_native_layout: bool = False
              ) -> np.ndarray:
        """执行矩阵乘法运算
        
        Args:
            a: 左矩阵
            b: 右矩阵
            out_type: 输出类型
            const_a: 是否将a视为常量矩阵(True时只在首次更新)
            const_b: 是否将b视为常量矩阵(True时只在首次更新)
            b_native_layout: 是否使用native layout存储矩阵B
            
        Returns:
            结果矩阵
        """
        # 检查输入
        if self.perf_debug:
            total_start = time.perf_counter()
            prep_start = time.perf_counter()
            
        # 检查输入和类型转换
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Input arrays must be 2-dimensional")
            
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Cannot multiply arrays with shapes {a.shape} and {b.shape}")
            
        # 转换为FP16并发出警告
        M, K = a.shape
        K, N = b.shape

        if self.perf_debug:
            prep_time = (time.perf_counter() - prep_start) * 1000
            init_start = time.perf_counter()
            
        # 设置B矩阵的layout
        self._b_layout = RKNNMatMulLayout.RKNN_MM_LAYOUT_NATIVE if b_native_layout else RKNNMatMulLayout.RKNN_MM_LAYOUT_NORM
        
        # 检查是否需要重新初始化
        need_init = (self._last_shape != (M, K, N) or 
                    self._out_type != out_type or
                    self.ctx.value == 0)
                    
        if need_init:
            if self.ctx.value != 0:
                # print(f"\033[93mWarning: Reinitializing matmul context for shape {M}x{K}x{N} and output type {out_type}\033[0m")
                # import traceback
                # print("调用栈:")
                # for line in traceback.format_stack()[:-1]:
                #     print(line.strip())
                pass
            self._init_context(M, K, N, out_type)
            
        if self.perf_debug:
            init_time = (time.perf_counter() - init_start) * 1000
            mem_start = time.perf_counter()
            
        # 更新内存
        if need_init or not const_a:
            if a.dtype != np.float16:
                # print(f"\033[93mWarning: Input matrix A dtype {a.dtype} will be converted to float16. "
                #     f"For better performance, please use float16 input directly.\033[0m")
                # import traceback
                # print("调用栈:")
                # for line in traceback.format_stack()[:-1]:
                #     print(line.strip())
                a = a.astype(np.float16)
            if not a.flags.c_contiguous:
                print(f"\033[93mWarning: Input matrix A is not C-contiguous. Copying to C-contiguous memory.\033[0m")
                a = a.copy(order='C')
            self._a_mem.update_from_numpy(a)
            ret = _lib.rknn_matmul_set_io_mem(self.ctx, self._a_mem.ptr, ctypes.byref(self._io_attr.A))
            if ret < 0:
                raise RuntimeError(f"Failed to set input A memory: {ret}")
            
        if need_init or not const_b:
            if b.dtype != np.float16:
                # print(f"\033[93mWarning: Input matrix B dtype {b.dtype} will be converted to float16. "
                #     f"For better performance, please use float16 input directly.\033[0m")
                b = b.astype(np.float16)
            if not b.flags.c_contiguous:
                print(f"\033[93mWarning: Input matrix B is not C-contiguous. Copying to C-contiguous memory.\033[0m")
                b = b.copy(order='C')
                pass
            if b_native_layout:
                layout_start = time.perf_counter()
                b_native = self._convert_b_to_native_layout(b)
                if self.perf_debug:
                    layout_time = (time.perf_counter() - layout_start) * 1000
                    print(f"Layout conversion time: {layout_time:.2f} ms")
                self._b_mem.update_from_numpy(b_native)
            else:
                self._b_mem.update_from_numpy(b)
            ret = _lib.rknn_matmul_set_io_mem(self.ctx, self._b_mem.ptr, ctypes.byref(self._io_attr.B))
            if ret < 0:
                raise RuntimeError(f"Failed to set input B memory: {ret}")
            
        ret = _lib.rknn_matmul_set_io_mem(self.ctx, self._c_mem.ptr, ctypes.byref(self._io_attr.C))
        if ret < 0:
            raise RuntimeError(f"Failed to set output C memory: {ret}")
            
        if self.perf_debug:
            mem_time = (time.perf_counter() - mem_start) * 1000
            compute_start = time.perf_counter()
            
        # 执行计算
        ret = _lib.rknn_matmul_run(self.ctx)
        
        if self.perf_debug:
            compute_time = (time.perf_counter() - compute_start) * 1000
            result_start = time.perf_counter()
            
        if ret < 0:
            raise RuntimeError(f"Failed to run matmul: {ret}")
            
        # 获取结果
        result = self._c_mem.to_numpy(shape=(M, N), 
                                  dtype=np.float32 if out_type == RKNNTensorType.RKNN_TENSOR_FLOAT32 else np.float16)
                              
        if self.perf_debug:
            result_time = (time.perf_counter() - result_start) * 1000
            total_time = (time.perf_counter() - total_start) * 1000
            
            # 计算GFLOPS
            flops = 2 * M * N * K
            gflops = (flops / (compute_time / 1000)) / 1e9
            
            print(f"\n\nPerformance Breakdown:")
            print(f"matmul shape: {M}x{K}x{N}")
            print(f"{'Stage':20} {'Time (ms)':>10}")
            print("-" * 32)
            print(f"{'Data preparation':20} {prep_time:>10.2f}")
            print(f"{'Context init':20} {init_time:>10.2f}")
            print(f"{'Memory operations':20} {mem_time:>10.2f}")
            print(f"{'Computation':20} {compute_time:>10.2f}")
            print(f"{'Result retrieval':20} {result_time:>10.2f}")
            print(f"{'Total time':20} {total_time:>10.2f}")
            print(f"\nCompute Performance: {gflops:.2f} GFLOPS")
            print("-" * 50)
            
        return result
    
    def matmul(self, 
           a: np.ndarray, 
           b: np.ndarray,
           out_type: RKNNTensorType = RKNNTensorType.RKNN_TENSOR_FLOAT32,
           const_a: bool = False,
           const_b: bool = False,
           b_native_layout: bool = False
          ) -> np.ndarray:
        """执行矩阵乘法运算，支持2D和3D输入
        
        Args:
            a: 左矩阵，形状为(..., M, K)
            b: 右矩阵，形状为(..., K, N)
            out_type: 输出类型
            const_a: 是否将a视为常量矩阵
            const_b: 是否将b视为常量矩阵
            b_native_layout: 是否使用native layout存储矩阵B
                
        Returns:
            结果矩阵，形状为(..., M, N)
        """
        if self.perf_debug:
            total_start = time.perf_counter()
        
        # 如果是2D输入，直接调用2D实现
        if a.ndim == 2 and b.ndim == 2:
            return self.matmul_2d(a, b, out_type, const_a, const_b, b_native_layout)
        
        # 检查输入维度
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("Input arrays must be at least 2-dimensional")
        
        # 获取最后两个维度的形状
        M, K1 = a.shape[-2:]
        K2, N = b.shape[-2:]
        
        if K1 != K2:
            raise ValueError(f"Cannot multiply arrays with shapes {a.shape} and {b.shape}")
        
        # 处理广播
        # 获取除最后两维外的形状
        a_batch_shape = a.shape[:-2]
        b_batch_shape = b.shape[:-2]
        
        try:
            # 计算广播后的batch形状
            batch_shape = np.broadcast_shapes(a_batch_shape, b_batch_shape)
        except ValueError as e:
            raise ValueError(f"Cannot broadcast shapes {a.shape} and {b.shape}") from e
        
        # 广播输入数组
        if a_batch_shape != batch_shape:
            a = np.broadcast_to(a, batch_shape + (M, K1))
        if b_batch_shape != batch_shape:
            b = np.broadcast_to(b, batch_shape + (K2, N))
        
        # 计算总的batch大小
        batch_size = np.prod(batch_shape) if batch_shape else 1
        
        # 重塑为2D进行计算
        a_2d = a.reshape(-1, M, K1)
        b_2d = b.reshape(-1, K2, N)
        
        # 创建输出数组
        out_dtype = np.float32 if out_type == RKNNTensorType.RKNN_TENSOR_FLOAT32 else np.float16
        result = np.empty(batch_shape + (M, N), dtype=out_dtype)
        result_2d = result.reshape(-1, M, N)
        
        # 对每个batch执行矩阵乘法
        for i in range(batch_size):
            result_2d[i] = self.matmul_2d(
                a_2d[i], 
                b_2d[i],
                out_type=out_type,
                const_a=const_a,
                const_b=const_b,
                b_native_layout=b_native_layout
            )
        
        if self.perf_debug:
            total_time = (time.perf_counter() - total_start) * 1000
            print(f"\nTotal batch processing time: {total_time:.2f} ms")
            print(f"Average time per batch: {total_time/batch_size:.2f} ms")
            print("-" * 50)
        
        return result
