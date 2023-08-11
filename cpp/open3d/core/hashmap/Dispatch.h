// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Dtype.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/MiniVec.h"

#define INSTANTIATE_TYPES(DTYPE, DIM)                \
    using key_t = utility::MiniVec<DTYPE, DIM>;      \
    using hash_t = utility::MiniVecHash<DTYPE, DIM>; \
    using eq_t = utility::MiniVecEq<DTYPE, DIM>;

#define DIM_SWITCHER(DTYPE, DIM, ...)                                    \
    if (DIM == 1) {                                                      \
        INSTANTIATE_TYPES(DTYPE, 1)                                      \
        return __VA_ARGS__();                                            \
    } else if (DIM == 2) {                                               \
        INSTANTIATE_TYPES(DTYPE, 2)                                      \
        return __VA_ARGS__();                                            \
    } else if (DIM == 3) {                                               \
        INSTANTIATE_TYPES(DTYPE, 3)                                      \
        return __VA_ARGS__();                                            \
    } else if (DIM == 4) {                                               \
        INSTANTIATE_TYPES(DTYPE, 4)                                      \
        return __VA_ARGS__();                                            \
    } else if (DIM == 5) {                                               \
        INSTANTIATE_TYPES(DTYPE, 5)                                      \
        return __VA_ARGS__();                                            \
    } else if (DIM == 6) {                                               \
        INSTANTIATE_TYPES(DTYPE, 6)                                      \
        return __VA_ARGS__();                                            \
    } else {                                                             \
        utility::LogError(                                               \
                "Unsupported dim {}, please modify {} and compile from " \
                "source",                                                \
                DIM, __FILE__);                                          \
    }

// TODO: dispatch more combinations.
#define DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(DTYPE, DIM, ...)                   \
    [&] {                                                                     \
        if (DTYPE == open3d::core::Int64) {                                   \
            DIM_SWITCHER(int64_t, DIM, __VA_ARGS__)                           \
        } else if (DTYPE == open3d::core::Int32) {                            \
            DIM_SWITCHER(int, DIM, __VA_ARGS__)                               \
        } else if (DTYPE == open3d::core::Int16) {                            \
            DIM_SWITCHER(short, DIM, __VA_ARGS__)                             \
        } else {                                                              \
            utility::LogError(                                                \
                    "Unsupported dtype {}, please use integer types (Int64, " \
                    "Int32, Int16).",                                         \
                    DTYPE.ToString());                                        \
        }                                                                     \
    }()

#ifdef __CUDACC__
// Reinterpret hash maps' void* value arrays as CUDA primitive types arrays, to
// avoid slow memcpy or byte-by-byte copy in kernels.
// Not used in the CPU version since memcpy is relatively fast on CPU.
#define DISPATCH_DIVISOR_SIZE_TO_BLOCK_T(DIVISOR, ...) \
    [&] {                                              \
        if (DIVISOR == 16) {                           \
            using block_t = int4;                      \
            return __VA_ARGS__();                      \
        } else if (DIVISOR == 12) {                    \
            using block_t = int3;                      \
            return __VA_ARGS__();                      \
        } else if (DIVISOR == 8) {                     \
            using block_t = int2;                      \
            return __VA_ARGS__();                      \
        } else if (DIVISOR == 4) {                     \
            using block_t = int;                       \
            return __VA_ARGS__();                      \
        } else if (DIVISOR == 2) {                     \
            using block_t = int16_t;                   \
            return __VA_ARGS__();                      \
        } else {                                       \
            using block_t = uint8_t;                   \
            return __VA_ARGS__();                      \
        }                                              \
    }()
#endif

namespace open3d {
namespace utility {

template <typename T, int N>
struct MiniVecHash {
public:
    OPEN3D_HOST_DEVICE uint64_t operator()(const MiniVec<T, N>& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int i = 0; i < N; ++i) {
            hash ^= static_cast<uint64_t>(key[i]);
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

template <typename T, int N>
struct MiniVecEq {
public:
    OPEN3D_HOST_DEVICE bool operator()(const MiniVec<T, N>& lhs,
                                       const MiniVec<T, N>& rhs) const {
        bool is_equal = true;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int i = 0; i < N; ++i) {
            is_equal = is_equal && (lhs[i] == rhs[i]);
        }
        return is_equal;
    }
};

}  // namespace utility
}  // namespace open3d
