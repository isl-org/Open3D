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
    \brief Implements the epilogue phase of the GEMM kernel that efficiently updates global memory
           with the computed matrix product.
*/

#pragma once

// clang-format off

#include "cutlass/zip_fragment.h"
#include "cutlass/zip_tile_iterator.h"
#include "cutlass/util/complex.h"
#include "cutlass/gemm/volta884_gemm_epilogue_traits.h"
#include "cutlass/gemm/scalar_or_pointer.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Adapter for linera scaling
template <typename Scalar_>
struct SplitComplexLinearScaling {

  /// Underlying real-valued scalar type
  typedef Scalar_ Scalar;

  /// Complex data type
  typedef platform::complex<Scalar> Complex;

  /// Parameters
  struct Params {

    /// Alpha
    Complex alpha;

    /// Beta
    Complex beta;

    //
    // Methods
    //

    // Constructor
    CUTLASS_HOST_DEVICE
    Params(): alpha(0, 0), beta(0, 0) {}

    // Constructor
    CUTLASS_HOST_DEVICE
    Params(Complex const & _alpha, Complex const & _beta) : alpha(_alpha), beta(_beta) {}

    /// Initialize the parameters
    CUTLASS_HOST_DEVICE
    int initialize(Complex const & _alpha, Complex const & _beta) {
      alpha = _alpha;
      beta = _beta;
      return 0;
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_HOST_DEVICE
  SplitComplexLinearScaling() { }

  /// Ctor.
  CUTLASS_HOST_DEVICE
  SplitComplexLinearScaling(Params const& _params) : params(_params) {}

  /// Method to determine whether the source accumulator matrix C is ever needed.
  CUTLASS_DEVICE
  bool source_required() const {
    return !is_zero(params.beta.real()) || !is_zero(params.beta.imag());
  }

  /// Evaluate the functor.
  template <typename FragmentA, typename FragmentC>
  CUTLASS_HOST_DEVICE
  void evaluate(FragmentA const& accum, FragmentC & output) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentA::First::kElements; ++i) {

      // Zip together split-complex accumulator representation for complex arithmetic
      Complex result = params.alpha * Complex(accum.first[i], accum.second[i]);

      output.first[i] = result.real();
      output.second[i] = result.imag();
    }
  }

  /// Evaluate the functor.
  template <typename FragmentA, typename FragmentC>
  CUTLASS_HOST_DEVICE
  void evaluate(FragmentA const& accum, FragmentC const& old, FragmentC& output) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentA::First::kElements; ++i) {

      // Zip together split-complex representations for complex arithmetic
      Complex source(old.first[i], old.second[i]);

      Complex result = params.alpha * Complex(accum.first[i], accum.second[i]) + params.beta * source;

      output.first[i] = result.real();
      output.second[i] = result.imag();
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Functor to compute linear combination of fragments. This is intended to support passing scalars
/// either by value from the host or by reference to device-side scalar elements. This is inspired
/// by cuBLAS's device pointer mode.
template <typename Scalar_ >
struct SplitComplexLinearScalingDevicePtr : public SplitComplexLinearScaling<Scalar_> {

  /// Linear Scaling class used
  typedef SplitComplexLinearScaling<Scalar_> Base;

  /// Underlying real-valued scalar type
  typedef typename Base::Scalar Scalar;

  /// Complex data type
  typedef platform::complex<Scalar> Complex;

  /// The parameters.
  class Params  {
  private:
    /// Alpha scalar
    detail::ScalarOrPointer<Complex> alpha_;

    /// Beta scalar
    detail::ScalarOrPointer<Complex> beta_;

  public:
    //
    // Methods
    //

    // Constructor
    CUTLASS_HOST_DEVICE
    Params() {}

    // Constructor
    CUTLASS_HOST_DEVICE
    Params(
      Complex alpha,
      Complex beta
    ):
      alpha_(alpha),
      beta_(beta) {}

    // Constructor
    CUTLASS_HOST_DEVICE
    Params(
      Complex const *alpha_ptr,
      Complex const *beta_ptr
    ):
      alpha_(alpha_ptr),
      beta_(alpha_ptr) {}

    /// Initialize the parameters
    CUTLASS_HOST_DEVICE int initialize(
      Complex alpha,
      Complex beta) {

      alpha_ = alpha;
      beta_ = beta;

      return 0;
    }

    /// Initialize the parameters
    CUTLASS_HOST_DEVICE int initialize(
      Complex const *alpha,
      Complex const *beta) {

      alpha_ = alpha;
      beta_= beta;

      return 0;
    }

    /// Initialize the parameters.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {

      alpha_ = desc.alpha;
      beta_ = desc.beta;

      return 0;
    }

    /// Gets the alpha scalar
    CUTLASS_HOST_DEVICE
    Complex alpha() const {
      return alpha_;
    }

    /// Gets the beta scalar
    CUTLASS_HOST_DEVICE
    Complex beta() const {
      return beta_;
    }
  };

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_HOST_DEVICE SplitComplexLinearScalingDevicePtr(Params const& _params) {
    this->params.alpha = _params.alpha();
    this->params.beta = _params.beta();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace cutlass
