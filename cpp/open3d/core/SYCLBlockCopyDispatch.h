// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace open3d {
namespace core {

/// Largest aligned block size (16, 12, 8, 4, 2, or 1 bytes) that divides
/// `object_byte_size`, for vectorized trivial object copies on SYCL.
inline int64_t GetLargestAlignedObjectBlockSize(int64_t object_byte_size) {
    static constexpr int64_t kDivisors[] = {16, 12, 8, 4, 2, 1};
    for (int64_t divisor : kDivisors) {
        if (object_byte_size % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

// Reinterpret object byte arrays as sycl::vec / scalar block arrays so the SYCL
// compiler can emit wide vector load/stores instead of byte loops.
// Mirrors DISPATCH_DIVISOR_SIZE_TO_BLOCK_T in hashmap/Dispatch.h (CUDA int4/…).
//
// Must be a macro (not a template) so that `using block_t = ...` creates a
// type alias local to the calling scope.
#define DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL(DIVISOR, ...) \
    [&] {                                                   \
        if (DIVISOR == 16) {                                \
            using block_t = sycl::vec<uint32_t, 4>;         \
            return __VA_ARGS__();                           \
        } else if (DIVISOR == 12) {                         \
            using block_t = sycl::vec<uint32_t, 3>;         \
            return __VA_ARGS__();                           \
        } else if (DIVISOR == 8) {                          \
            using block_t = sycl::vec<uint32_t, 2>;         \
            return __VA_ARGS__();                           \
        } else if (DIVISOR == 4) {                          \
            using block_t = uint32_t;                       \
            return __VA_ARGS__();                           \
        } else if (DIVISOR == 2) {                          \
            using block_t = uint16_t;                       \
            return __VA_ARGS__();                           \
        } else {                                            \
            using block_t = uint8_t;                        \
            return __VA_ARGS__();                           \
        }                                                   \
    }()

}  // namespace core
}  // namespace open3d
