// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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

/// Largest entry in kBlockCopyDivisors that divides \p object_byte_size.
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

#ifdef SYCL_LANGUAGE_VERSION

// Reinterpret object byte arrays as sycl::vec / scalar block arrays so the SYCL
// compiler can emit wide vector load/stores instead of byte loops.
// Mirrors DISPATCH_DIVISOR_SIZE_TO_BLOCK_T in hashmap/Dispatch.h (CUDA).
//
// Must be a macro (not a template) so that `using block_t = ...` creates a
// type alias local to the calling scope.
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
