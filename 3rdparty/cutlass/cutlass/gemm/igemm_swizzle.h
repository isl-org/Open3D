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
    \brief Transposes a fragment of data containing packed 8-bit integer elements.
*/
#pragma once

#include "cutlass/fragment.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GlobalIterator_>
struct IgemmSwizzle {
  /// The global iterator.
  typedef GlobalIterator_ GlobalIterator;
  /// The source fragment.
  typedef typename GlobalIterator::Fragment Fragment;
  /// The shape of the source fragment.
  typedef typename GlobalIterator::FragmentShape FragmentShape;

  /// The source fragment.
  typedef Fragment InputFragment;
  /// The destination fragment.
  typedef Fragment OutputFragment;

  /// The src/dst must be int8 fragments.
  static_assert((platform::is_same<typename Fragment::Element, int8_t>::value), "Works on int8");

  /// The number of elements must be a multiple of 4.
  static_assert(FragmentShape::kH % 4 == 0 && ShapeCount<FragmentShape>::kWc % 4 == 0,
                "Not multiple of 4");

  /// Ctor.
  CUTLASS_DEVICE IgemmSwizzle() {}

  /// Transform a fragment.
  CUTLASS_DEVICE void transform(Fragment const& src, Fragment& dst) {

    // Expose src/dst as int arrays.
    int const* src_int = reinterpret_cast<int const*>(&src[0]);
    int* dst_int = reinterpret_cast<int*>(&dst[0]);

    // Transpose the data.
    for (int d = 0; d < FragmentShape::kD; ++d) {
      for (int h = 0; h < FragmentShape::kH / 4; ++h) {
        for (int w = 0; w < ShapeCount<FragmentShape>::kWc / 4; ++w) {
          int const i0 = d * (ShapeCount<FragmentShape>::kHwc / 4) +
                         (4 * h + 0) * (ShapeCount<FragmentShape>::kWc / 4) + w;
          int const i1 = d * (ShapeCount<FragmentShape>::kHwc / 4) +
                         (4 * h + 1) * (ShapeCount<FragmentShape>::kWc / 4) + w;
          int const i2 = d * (ShapeCount<FragmentShape>::kHwc / 4) +
                         (4 * h + 2) * (ShapeCount<FragmentShape>::kWc / 4) + w;
          int const i3 = d * (ShapeCount<FragmentShape>::kHwc / 4) +
                         (4 * h + 3) * (ShapeCount<FragmentShape>::kWc / 4) + w;

          int a0 = src_int[i0];
          int a1 = src_int[i1];
          int a2 = src_int[i2];
          int a3 = src_int[i3];

          // // DEBUG.
          // if (threadIdx.x == 0) {
          //     printf("a=0x%08x 0x%08x 0x%08x 0x%08x\n", a0, a1, a2, a3);
          // }

          int b0, b1, b2, b3, c0;
          asm volatile("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(b0) : "r"(a0), "r"(a1));
          asm volatile("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(c0) : "r"(a2), "r"(a3));
          asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b0) : "r"(b0), "r"(c0));

          asm volatile("prmt.b32 %0, %1, %2, 0x0051;" : "=r"(b1) : "r"(a0), "r"(a1));
          asm volatile("prmt.b32 %0, %1, %2, 0x0051;" : "=r"(c0) : "r"(a2), "r"(a3));
          asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b1) : "r"(b1), "r"(c0));

          asm volatile("prmt.b32 %0, %1, %2, 0x0062;" : "=r"(b2) : "r"(a0), "r"(a1));
          asm volatile("prmt.b32 %0, %1, %2, 0x0062;" : "=r"(c0) : "r"(a2), "r"(a3));
          asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b2) : "r"(b2), "r"(c0));

          asm volatile("prmt.b32 %0, %1, %2, 0x0073;" : "=r"(b3) : "r"(a0), "r"(a1));
          asm volatile("prmt.b32 %0, %1, %2, 0x0073;" : "=r"(c0) : "r"(a2), "r"(a3));
          asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(b3) : "r"(b3), "r"(c0));

          // // DEBUG.
          // if (threadIdx.x == 0) {
          //     printf("b=0x%08x 0x%08x 0x%08x 0x%08x\n", b0, b1, b2, b3);
          // }

          dst_int[i0] = b0;
          dst_int[i1] = b1;
          dst_int[i2] = b2;
          dst_int[i3] = b3;
        }
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
