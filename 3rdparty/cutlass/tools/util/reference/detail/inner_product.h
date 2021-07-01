/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Reference implementation for GEMM in host-side code.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/vector.h"

namespace cutlass {
namespace reference {
namespace detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Template function to compute an inner product.
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate with a
                            // host-only type
template <typename Atype, typename Btype, typename Ctype>
CUTLASS_HOST_DEVICE
Ctype inner_product(Atype a, Btype b, Ctype c) {
  return Ctype(a) * Ctype(b) + c;
}

#if defined(__clang__) && defined(__CUDA__)
__device__ __forceinline__ __half inner_product(__half a, __half b, __half c) {
  return a * b + c;
}
#endif

/// Specialization for matrix multiplication with binary operands
template <>
CUTLASS_HOST_DEVICE
int inner_product<Vector<bin1_t, 32>, Vector<bin1_t, 32>, int>(
    Vector<bin1_t, 32> a,
    Vector<bin1_t, 32> b,
    int c) {

  int accum = 0;
  for (int bit = 0; bit < 32; bit++) {
    accum += a[bit] ^ b[bit];
  }
  return accum + c;
}

/// Specialization for matrix multiplication with signed 4-bit integer operands
template <>
CUTLASS_HOST_DEVICE
int inner_product<Vector<int4_t, 8>, Vector<int4_t, 8>, int>(
    Vector<int4_t, 8> a,
    Vector<int4_t, 8> b,
    int c) {

  int accum = 0;
  for (int k = 0; k < 8; k++) {
    accum += a[k] * b[k];
  }
  return accum + c;
}

/// Specialization for matrix multiplication with unsigned 4-bit integer operands
template <>
CUTLASS_HOST_DEVICE
int inner_product<Vector<uint4_t, 8>, Vector<uint4_t, 8>, int>(
    Vector<uint4_t, 8> a,
    Vector<uint4_t, 8> b,
    int c) {

  int accum = 0;
  for (int k = 0; k < 8; k++) {
    accum += a[k] * b[k];
  }
  return accum + c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType>
struct Cast {
  // Default behavior: convert to the destination type
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  static DstType apply(SrcType src) { return static_cast<DstType>(src); };
};

template <>
struct Cast<float, int8_t> {
  CUTLASS_HOST_DEVICE
  static int8_t apply(float src) {
    // Clamp to the range of signed 8-bit integers.
    return static_cast<int8_t>(fmaxf(-128.f, fminf(127.f, src)));
  };
};

template <>
struct Cast<float, uint8_t> {
  CUTLASS_HOST_DEVICE
  static uint8_t apply(float src) {
    // Clamp to the range of signed 8-bit integers.
    return static_cast<uint8_t>(fmaxf(0.f, fminf(255.f, src)));
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail
} // namespace reference
} // namespace cutlass
