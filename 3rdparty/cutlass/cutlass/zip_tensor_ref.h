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
    \brief Defines a structure containing a pair of TensorRef-like objects
*/
#pragma once

#include "cutlass/coord.h"
#include "cutlass/tensor_ref.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename First_, typename Second_>
struct ZipTensorRef {
  /// First tensor ref
  typedef First_ First;

  /// Second tensor ref
  typedef Second_ Second;

  //
  // Data members
  //

  /// First TensorRef
  First first;

  /// Second TensorRef
  Second second;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ZipTensorRef() {}

  CUTLASS_HOST_DEVICE
  ZipTensorRef(First const& _first, Second const& _second) : first(_first), second(_second) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructs a ZipTensorRef
template <typename First, typename Second>
CUTLASS_HOST_DEVICE
ZipTensorRef<First, Second> make_ZipTensorRef(First const &first, Second const &second) {
  return ZipTensorRef<First, Second>(first, second);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Any simple way to do so?
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename First_, typename Second_, typename Third_>
struct Zip3TensorRef {
  /// First tensor ref
  typedef First_ First;

  /// Second tensor ref
  typedef Second_ Second;

  /// Third tensor ref
  typedef Third_ Third;

  //
  // Data members
  //

  /// First TensorRef
  First first;

  /// Second TensorRef
  Second second;

  /// Third TensorRef
  Third third;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Zip3TensorRef() {}

  CUTLASS_HOST_DEVICE
  Zip3TensorRef(First const& _first, Second const& _second, Third const& _third) : 
    first(_first), second(_second), third(_third) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructs a ZipTensorRef
template <typename First, typename Second, typename Third>
CUTLASS_HOST_DEVICE
Zip3TensorRef<First, Second, Third> make_Zip3TensorRef(First const &first, 
                                                       Second const &second,
                                                       Third const &third) {
  return Zip3TensorRef<First, Second, Third>(first, second, third);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
