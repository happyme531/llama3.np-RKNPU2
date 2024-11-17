#include <cmath>
#include <cstdint>
#include <cstring>


// 通用的layout转换函数
template <typename Ti, typename To>
void norm_layout_to_native_layout(Ti *src, To *dst, int32_t K, int32_t N, int32_t subN, int32_t subK) {
    int N_remain = (int)std::ceil(N * 1.0f / subN);
    int K_remain = (int)std::ceil(K * 1.0f / subK);
    
    for (int i = 0; i < N_remain; i++) {
        for (int j = 0; j < K_remain; j++) {
            for (int n = 0; n < subN; n++) {
                int ni = i * subN + n;
                for (int k = 0; k < subK; k++) {
                    int ki = j * subK + k;
                    if (ki < K && ni < N) {
                        dst[((i * K_remain + j) * subN + n) * subK + k] = src[ki * N + ni];
                    } else {
                        dst[((i * K_remain + j) * subN + n) * subK + k] = 0;
                    }
                }
            }
        }
    }
}


extern "C" {
// FP16版本的layout转换
int fp16_norm_to_native_layout(void *src, void *dst, int32_t K, int32_t N) {
    // FP16使用16x32的block大小
    const int subN = 16;
    const int subK = 32;
    
    try {
        norm_layout_to_native_layout(
            static_cast<uint16_t*>(src),  // FP16用uint16_t表示
            static_cast<uint16_t*>(dst),
            K, N, subN, subK
        );
        return 0;
    } catch (...) {
        return -1;
    }
}

// INT8版本的layout转换 
int int8_norm_to_native_layout(void *src, void *dst, int32_t K, int32_t N) {
    // INT8使用32x32的block大小
    const int subN = 32;
    const int subK = 32;
    
    try {
        norm_layout_to_native_layout(
            static_cast<int8_t*>(src),
            static_cast<int8_t*>(dst),
            K, N, subN, subK
        );
        return 0;
    } catch (...) {
        return -1;
    }
}

// INT4版本的layout转换
int int4_norm_to_native_layout(void *src, void *dst, int32_t K, int32_t N) {
    // INT4使用64x32的block大小
    const int subN = 64;
    const int subK = 32;
    
    try {
        norm_layout_to_native_layout(
            static_cast<int8_t*>(src),  // INT4实际存储在int8中
            static_cast<int8_t*>(dst),
            K, N, subN, subK
        );
        return 0;
    } catch (...) {
        return -1;
    }
}

} // extern "C"
