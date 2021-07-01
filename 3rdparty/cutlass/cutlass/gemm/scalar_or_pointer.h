
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

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Helper class defines an object  which operates as either a scalar or a pointer. If the pointer
/// is non-null, it is dereferenced when the object is accessed.
template <typename Scalar_>
class ScalarOrPointer {
public:
  /// Underlying scalar type
  typedef Scalar_ Scalar;

private:
  //
  // Data members
  //

  /// Scalar value
  Scalar scalar;

  /// Pointer to use if non null
  Scalar const *ptr;

public:

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  ScalarOrPointer(): scalar(0), ptr(nullptr) {}

  /// Object behaves as a scalar
  CUTLASS_HOST_DEVICE
  ScalarOrPointer(Scalar const &val): scalar(val), ptr(nullptr) {}

  /// Object behaves as a scalar
  CUTLASS_HOST_DEVICE
  ScalarOrPointer(Scalar const *ptr_): scalar(0), ptr(ptr_) {}

  /// Returns true if is pointer
  CUTLASS_HOST_DEVICE
  bool is_pointer() const {
    return bool(ptr);
  }

  /// Gets the pointer value
  CUTLASS_HOST_DEVICE
  Scalar const *get_ptr() const {
    return ptr;
  }

  /// Gets the pointer value
  CUTLASS_HOST_DEVICE
  Scalar get_scalar() const {
    return scalar;
  }

  /// Assigns to a scalar and sets pointer to nullptr
  CUTLASS_HOST_DEVICE
  ScalarOrPointer &operator=(Scalar const &scalar_) {
    scalar = scalar_;
    ptr = nullptr;
    return *this;
  }

  /// Assigns to a pointer value
  CUTLASS_HOST_DEVICE
  ScalarOrPointer &operator=(Scalar const *ptr_) {
    ptr = ptr_;
    return *this;
  }

  /// Access the element
  CUTLASS_HOST_DEVICE
  Scalar get() const {
    if (ptr) {
      return *ptr;
    }
    return scalar;
  }

  /// Accesses the element
  CUTLASS_HOST_DEVICE
  operator Scalar() const {
    return get();
  }
};

} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
