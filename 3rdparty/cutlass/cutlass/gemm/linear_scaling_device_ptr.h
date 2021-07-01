
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/scalar_or_pointer.h"
#include "cutlass/gemm/linear_scaling.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Functor to compute linear combination of fragments. This is intended to support passing scalars
/// either by value from the host or by reference to device-side scalar elements. This is inspired
/// by cuBLAS's device pointer mode.
template <typename Scalar_, typename FragmentMultiplyAdd_ = FragmentMultiplyAdd<Scalar_, Scalar_> >
struct LinearScalingDevicePtr : public LinearScaling<Scalar_, FragmentMultiplyAdd_> {

  /// Linear Scaling class used
  typedef LinearScaling<Scalar_, FragmentMultiplyAdd_> Base;

  // The scalar.
  typedef typename Base::Scalar Scalar;

  /// The parameters.
  class Params  {
  private:
    /// Alpha scalar
    detail::ScalarOrPointer<Scalar> alpha_;

    /// Beta sclaar
    detail::ScalarOrPointer<Scalar> beta_;

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
      Scalar alpha,
      Scalar beta
    ):
      alpha_(alpha),
      beta_(beta) {}

    // Constructor
    CUTLASS_HOST_DEVICE
    Params(
      Scalar const *alpha_ptr,
      Scalar const *beta_ptr
    ):
      alpha_(alpha_ptr),
      beta_(alpha_ptr) {}

    /// Initialize the parameters
    CUTLASS_HOST_DEVICE int initialize(
      Scalar alpha,
      Scalar beta) {

      alpha_ = alpha;
      beta_ = beta;

      return 0;
    }

    /// Initialize the parameters
    CUTLASS_HOST_DEVICE int initialize(
      Scalar const *alpha,
      Scalar const *beta) {

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
    Scalar alpha() const {
      return alpha_;
    }

    /// Gets the beta scalar
    CUTLASS_HOST_DEVICE
    Scalar beta() const {
      return beta_;
    }
  };

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_HOST_DEVICE LinearScalingDevicePtr(Params const& _params) {
    this->params.alpha = _params.alpha();
    this->params.beta = _params.beta();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace cutlass
