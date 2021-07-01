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
    \brief Transposes a tile of 16b elements. Used by HGEMM to construct a K-strided layout in
      shared memory for multiplicands.
*/
#pragma once

#include <cuda_fp16.h>
#include "cutlass/fragment.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GlobalIterator_>
struct HgemmSwizzle {
  /// The global iterator.
  typedef GlobalIterator_ GlobalIterator;
  /// The source fragment.
  typedef typename GlobalIterator::Fragment Fragment;
  /// The shape of the source fragment.
  typedef typename GlobalIterator::FragmentShape FragmentShape;

  /// The input fragment.
  typedef Fragment InputFragment;
  /// The output fragment.
  typedef Fragment OutputFragment;

  /// The src/dst must be half fragments.
  static_assert((platform::is_same<typename Fragment::Element, half>::value), "Works on half");

  /// The number of elements must be a multiple of 2.
  static_assert(FragmentShape::kH == 2 && ShapeCount<FragmentShape>::kWc == 2, "Not multiple of 2");

  /// Ctor.
  CUTLASS_DEVICE HgemmSwizzle() {}

  /// Transform a fragment.
  CUTLASS_DEVICE void transform(Fragment const& src, Fragment& dst) {
    // Expose src/dst as int arrays.
    int const* src_int = reinterpret_cast<int const*>(&src[0]);
    int* dst_int = reinterpret_cast<int*>(&dst[0]);

    // Transpose the data.
    for (int d = 0; d < FragmentShape::kD; ++d) {
      // The indices to read two consecutive "rows".
      int const i0 = 2 * d + 0;
      int const i1 = 2 * d + 1;

      int a0 = src_int[i0];
      int a1 = src_int[i1];

      int b0, b1;
      asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b0) : "r"(a0), "r"(a1));
      asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(b1) : "r"(a0), "r"(a1));

      // The indices to store with "strides".
      int const j0 = 0 * (ShapeCount<FragmentShape>::kDhw / 2) + d;
      int const j1 = 1 * (ShapeCount<FragmentShape>::kDhw / 2) + d;

      dst_int[j0] = b0;
      dst_int[j1] = b1;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
