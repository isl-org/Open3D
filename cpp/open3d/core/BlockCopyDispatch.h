// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file BlockCopyDispatch.h
/// \brief Vectorized trivial-object copy block sizes (CUDA and SYCL).
///
/// \ref kBlockCopyDivisors selects the widest vector load/store type that
/// divides an object byte size. Used for hash-map SoA buffers, Dtype::Object
/// index get/set, and non-contiguous SYCL tensor copies of object dtypes.

#pragma once

#include <cstdint>

#ifdef SYCL_LANGUAGE_VERSION
#include <sycl/sycl.hpp>
#endif

namespace open3d {
namespace core {

// Vectorized trivial-object copy block sizes (bytes). Used for: HashMap SoA
// value buffers (VoxelBlockGrid blocks, etc.), Dtype::Object advanced index
// get/set, and non-contiguous SYCL tensor copy of object dtypes.
inline constexpr int64_t kBlockCopyDivisors[] = {64, 16, 12, 4, 1};

/// Largest entry in \ref kBlockCopyDivisors that divides \p object_byte_size.
inline int64_t GetLargestAlignedObjectBlockSize(int64_t object_byte_size) {
    for (int64_t divisor : kBlockCopyDivisors) {
        if (object_byte_size % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

}  // namespace core
}  // namespace open3d

#ifdef __CUDACC__
#include <cuda_runtime.h>

// Reinterpret hash maps' void* value arrays as CUDA primitive type arrays to
// avoid slow memcpy or byte-by-byte copy in kernels. Not used on CPU (memcpy is
// fast enough). BlockCopy64 is at namespace scope because nvcc disallows
// types with no linkage as template arguments for __global__ instantiations.
struct BlockCopy64 {
    int4 v[4];
};

/// Pick a CUDA \c block_t type from \ref kBlockCopyDivisors (bytes per block).
#define DISPATCH_DIVISOR_SIZE_TO_BLOCK_T(DIVISOR, ...) \
    [&] {                                              \
        if (DIVISOR == 64) {                           \
            using block_t = BlockCopy64;               \
            return __VA_ARGS__();                      \
        } else if (DIVISOR == 16) {                    \
            using block_t = int4;                      \
            return __VA_ARGS__();                      \
        } else if (DIVISOR == 12) {                    \
            using block_t = int3;                      \
            return __VA_ARGS__();                      \
        } else if (DIVISOR == 4) {                     \
            using block_t = int;                       \
            return __VA_ARGS__();                      \
        } else {                                       \
            using block_t = uint8_t;                   \
            return __VA_ARGS__();                      \
        }                                              \
    }()
#endif  // __CUDACC__

#ifdef SYCL_LANGUAGE_VERSION

/// Reinterpret object byte arrays as \c sycl::vec / scalar block arrays for
/// wide vector loads/stores (mirrors CUDA \c DISPATCH_DIVISOR_SIZE_TO_BLOCK_T).
///
/// Must be a macro (not a template) so \c using block_t = ... is local to the
/// caller's scope.
#define DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL(DIVISOR, ...) \
    [&] {                                                   \
        if (DIVISOR == 64) {                                \
            using block_t = sycl::vec<uint32_t, 16>;        \
            return __VA_ARGS__();                           \
        } else if (DIVISOR == 16) {                         \
            using block_t = sycl::vec<uint32_t, 4>;         \
            return __VA_ARGS__();                           \
        } else if (DIVISOR == 12) {                         \
            using block_t = sycl::vec<uint32_t, 3>;         \
            return __VA_ARGS__();                           \
        } else if (DIVISOR == 4) {                          \
            using block_t = uint32_t;                       \
            return __VA_ARGS__();                           \
        } else {                                            \
            using block_t = uint8_t;                        \
            return __VA_ARGS__();                           \
        }                                                   \
    }()

#endif  // SYCL_LANGUAGE_VERSION
