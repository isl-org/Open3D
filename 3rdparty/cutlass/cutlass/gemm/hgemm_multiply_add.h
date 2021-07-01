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
    \brief Specialization implementing multiply-add operation on half-precision floating point
   fragments.
*/
#pragma once

#include "cutlass/fragment.h"
#include "cutlass/gemm/thread_multiply_add.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Template performing matrix multiply-add operation within a thread
template <typename ThreadGemmShape_, typename ThreadsPerWarp_>
struct ThreadMultiplyAdd<ThreadGemmShape_, ThreadsPerWarp_, half, half, half> {
  /// The shape of the instruction.
  typedef Shape<1, 1, 2, 1> InstructionShape;
  /// The number of accumulators per thread.
  typedef ThreadGemmShape_ ThreadGemmShape;
  /// Aliased for compatibility. Will be removed for CUTLASS v2.0.
  typedef ThreadGemmShape AccumulatorsPerThread;
  /// The number of threads per warp.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of accumulators per warp.
  typedef typename ShapeMul<ThreadGemmShape, ThreadsPerWarp>::Shape AccumulatorsPerWarp;
  /// The type for A.
  typedef half ScalarA;
  /// The fragment for A.
  typedef Fragment<ScalarA, AccumulatorsPerThread::kW> FragmentA;
  /// The type for B.
  typedef half ScalarB;
  /// The fragment for B.
  typedef Fragment<ScalarB, AccumulatorsPerThread::kH> FragmentB;
  /// The type for C and D.
  typedef half ScalarC;
  /// The accumulators.
  typedef Fragment<half, AccumulatorsPerThread::kH * AccumulatorsPerThread::kW> Accumulators;

  /// Make sure there's an even number of elements in both dimensions.
  static_assert(AccumulatorsPerThread::kH % 2 == 0, "Invalid size");
  static_assert(AccumulatorsPerThread::kW % 2 == 0, "Invalid size");
  static_assert(AccumulatorsPerThread::kH >= 2 && AccumulatorsPerThread::kW >= 2,
    "HGEMM expects at least 2x2 accmulator tiles per thread.");

  /// Ctor.
  CUTLASS_DEVICE ThreadMultiplyAdd() {}

  /// Multiply : d = a*b + c.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 530
    // The inputs.
    __half2 const* a_half2 = reinterpret_cast<__half2 const*>(&a[0]);
    __half2 const* b_half2 = reinterpret_cast<__half2 const*>(&b[0]);
    __half2 const* c_half2 = reinterpret_cast<__half2 const*>(&c[0]);

    // The output.
    __half2* d_half2 = reinterpret_cast<__half2*>(&d[0]);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < AccumulatorsPerThread::kH / 2; ++j) {

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < AccumulatorsPerThread::kW / 2; ++i) {
        // The offsets in the output fragment.
        int const k0 = (2 * j + 0) * (AccumulatorsPerThread::kW / 2) + i;
        int const k1 = (2 * j + 1) * (AccumulatorsPerThread::kW / 2) + i;

        // Compute the product a[i] * b[j].low.
        d_half2[k0] = __hfma2(a_half2[i], __low2half2(b_half2[j]), c_half2[k0]);
        // Compute the product a[i] * b[j].high.
        d_half2[k1] = __hfma2(a_half2[i], __high2half2(b_half2[j]), c_half2[k1]);
      }
    }
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
