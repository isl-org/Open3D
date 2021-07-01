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
    \brief Implements the BLAS linear scaling function alpha*AB + beta*C
*/
#pragma once

#include "cutlass/fragment_multiply_add.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
CUTLASS_DEVICE bool is_zero(T x) {
  return x == T(0);
}

#if !defined(__CUDACC_RTC__) || defined(CUTLASS_NVRTC_HAS_FP16)
CUTLASS_DEVICE bool is_zero(half x) { return reinterpret_cast<int16_t&>(x) == int16_t(0); }
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Functor to compute linear combination of fragments
template <typename Scalar_, typename FragmentMultiplyAdd_ = FragmentMultiplyAdd<Scalar_, Scalar_> >
struct LinearScaling {
  // The scalar.
  typedef Scalar_ Scalar;
  // The accumulator Type
  typedef typename FragmentMultiplyAdd_::ScalarAccum ScalarAccum;
  // The adapater.
  typedef FragmentMultiplyAdd_ FragmentMultiplyAdd;

  /// The parameters.
  struct Params {
    /// The alpha/beta scaling params.
    Scalar alpha, beta;

    //
    // Methods
    //

    // Constructor
    CUTLASS_HOST_DEVICE
    Params(Scalar _alpha = 0.0f, Scalar _beta = 0.0f)
        : alpha(_alpha), beta(_beta) {}

    /// Initialize the parameters
    CUTLASS_HOST_DEVICE int initialize(Scalar _alpha, Scalar _beta) {
      alpha = _alpha;
      beta = _beta;
      return 0;
    }

    /// Initialize the parameters.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {
      alpha = desc.alpha;
      beta = desc.beta;
      return 0;
    }
  };

  //
  // Data members
  //

  Params params;

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_DEVICE LinearScaling() { }

  /// Ctor.
  CUTLASS_DEVICE LinearScaling(Params const& _params) : params(_params) {}

  /// Method to determine whether the source accumulator matrix C is ever needed. This method
  /// may always safely return true, though better performance is possible if the source accumulator
  /// matrix is never loaded unnecessarily.
  CUTLASS_DEVICE
  bool source_required() const {
    return !is_zero(params.beta);
  }

  /// Evaluate the functor.
  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_& output) {
    FragmentMultiplyAdd mad;
    mad.multiply(params.alpha, accum, output);

  }

  /// Evaluate the functor, without using fragment in the API
  template <typename ScalarAccum, typename ScalarOutput, int size>
  CUTLASS_DEVICE void evaluate(ScalarAccum const *accum, ScalarOutput *output) {
    Fragment<ScalarAccum, size> FragAccum;
    Fragment<ScalarOutput, size> FragOutput;
#pragma unroll
    for (int i = 0; i < size; i++) {
      FragAccum[i] = accum[i];
      FragOutput[i] = output[i];
    }
    evaluate(FragAccum, FragOutput);
#pragma unroll
    for (int i = 0; i < size; i++) {
      output[i] = FragOutput[i];
    }
  }

  /// Evaluate the functor.
  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_ const& old, FragmentB_& output) {
    FragmentMultiplyAdd mad;
    FragmentB_ tmp;
    mad.multiply(params.beta, old, tmp);
    mad.multiply_add(params.alpha, accum, tmp, output);
  }

  /// Evaluate the functor, without using fragment in the API
  template <typename ScalarAccum, typename ScalarOutput, int size>
  CUTLASS_DEVICE void evaluate(ScalarAccum const *accum, ScalarOutput const *old, ScalarOutput *output) {
    Fragment<ScalarAccum, size> FragAccum;
    Fragment<ScalarOutput, size> FragOutput;
    Fragment<ScalarOutput, size> FragOld;
#pragma unroll
    for (int i = 0; i < size; i++) {
      FragAccum[i] = accum[i];
      FragOutput[i] = output[i];
      FragOld[i] = old[i];
    }
    evaluate(FragAccum, FragOld, FragOutput);
#pragma unroll
    for (int i = 0; i < size; i++) {
      output[i] = FragOutput[i];
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
