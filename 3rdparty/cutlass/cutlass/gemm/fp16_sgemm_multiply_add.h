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
    \brief Template implementing matrix multiply-add operations on fragments.
*/
#pragma once

#include "cutlass/fragment.h"
#include "cutlass/gemm/thread_multiply_add.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Template performing matrix multiply-add operation within a thread
template <typename ThreadGemmShape_,
          typename ThreadsPerWarp_>
struct ThreadMultiplyAdd<ThreadGemmShape_, ThreadsPerWarp_, half, half, float> {
  /// The shape of the instruction.
  typedef Shape<1, 1, 1, 1> InstructionShape;
  /// The shape of a thread-leveel matrix multiply accumulate.
  typedef ThreadGemmShape_ ThreadGemmShape;
  /// Aliased to "AccumulatorsPerThread" for compatibility. Expect to be renamed in CUTLASS v2.0
  typedef ThreadGemmShape AccumulatorsPerThread;
  /// The number of threads per warp.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of accumulators per warp.
  typedef typename ShapeMul<ThreadGemmShape, ThreadsPerWarp>::Shape AccumulatorsPerWarp;
  /// The type for A. specialized to half
  typedef half ScalarA;
  /// The fragment for A.
  typedef Fragment<ScalarA, AccumulatorsPerThread::kW> FragmentA;
  /// The type for B. specialized to half
  typedef half ScalarB;
  /// The fragment for B.
  typedef Fragment<ScalarB, AccumulatorsPerThread::kH> FragmentB;
  /// The type for C and D. specialized to float
  typedef float ScalarC;
  /// The accumulators.
  typedef Fragment<ScalarC, AccumulatorsPerThread::kH * AccumulatorsPerThread::kW, 16> Accumulators;

  /// Ctor.
  CUTLASS_DEVICE ThreadMultiplyAdd() {}

  /// Multiply : d = a*b + c.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {

    for (int j = 0; j < AccumulatorsPerThread::kH; ++j) {
      for (int i = 0; i < AccumulatorsPerThread::kW; ++i) {

        d[j * AccumulatorsPerThread::kW + i] = static_cast<ScalarC>(a[i]) * static_cast<ScalarC>(b[j]) + c[j * AccumulatorsPerThread::kW + i];
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
