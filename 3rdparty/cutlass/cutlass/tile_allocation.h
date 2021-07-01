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
    \brief Defines a fragment based on a Shape<> template.
*/
#pragma once

#include "cutlass/shape.h"
#include "cutlass/fragment.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/zip_tensor_ref.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for storing a tile in memory and accessing it through a tensor ref
template <typename Scalar_, typename Shape_>
struct TileAllocation {
  //
  // Type definitions
  //

  /// Scalar element
  typedef Scalar_ Scalar;

  /// The actual storage (may differ from the scalar type)
  typedef typename StorageType<int(sizeof(Scalar))>::Type Storage;

  /// Size of the allocation in units of scalars
  typedef Shape_ Shape;

  /// Strides
  typedef typename ShapeStrides<Shape, 1>::Shape Strides;

  /// Defines the tensor reference for this allocation
  typedef TensorRef<Scalar const, 4> ConstTensorRef;

  /// Defines the tensor reference for this allocation
  typedef TensorRef<Scalar, 4> TensorRef;

  /// View of memory
  typedef TensorView<Scalar const, 4> ConstTensorView;

  /// View of memory
  typedef TensorView<Scalar, 4> TensorView;

  //
  // Data members
  //

  /// Storage
  Storage storage[Shape::kD][Shape::kH][Shape::kW][Shape::kC];

  //
  // Methods
  //

  /// Returns a pointer to the raw data
  CUTLASS_DEVICE
  Scalar *data() { return reinterpret_cast<Scalar *>(&storage[0][0][0][0]); }

  /// Returns a const pointer to the raw data
  CUTLASS_DEVICE
  Scalar const *data() const { return reinterpret_cast<Scalar const *>(&storage[0][0][0][0]); }

  /// Returns a TensorRef object pointing to the data
  CUTLASS_DEVICE
  TensorRef reference() {
    return TensorRef(data(), make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC));
  }

  /// Returns a TensorRef object pointing to the data
  CUTLASS_DEVICE
  ConstTensorRef reference() const {
    return ConstTensorRef(data(), make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC));
  }

  /// Returns a TensorView object pointing to the data
  CUTLASS_DEVICE
  TensorView view() {
    return TensorView(
      data(),
      make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC),
      make_Coord(Shape::kD, Shape::kH, Shape::kW, Shape::kC));
  }

  /// Returns a TensorView object pointing to the data
  CUTLASS_DEVICE
  ConstTensorView view() const {
    return TensorView(
      data(),
      make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC),
      make_Coord(Shape::kD, Shape::kH, Shape::kW, Shape::kC));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Manages a pair of tile allocations as if they are one allocation
template <typename First_, typename Second_>
struct ZipTileAllocation {
  //
  // Type definitions
  //

  /// First tensor allocation
  typedef First_ First;

  /// Second tensor allocation
  typedef Second_ Second;

  /// Defines the tensor reference for this allocation
  typedef ZipTensorRef<typename First::TensorRef, typename Second::TensorRef> TensorRef;

  /// Defines the tensor reference for this allocation
  typedef ZipTensorRef<typename First::ConstTensorRef, typename Second::ConstTensorRef>
      ConstTensorRef;

  //
  // Data members
  //

  /// First tensor allocation
  First first;

  /// Second tensor allocation
  Second second;

  //
  // Methods
  //

  /// Returns a TensorRef object pointing to the data
  CUTLASS_DEVICE
  TensorRef reference() { return TensorRef(first.reference(), second.reference()); }

  /// Returns a TensorRef object pointing to the data
  CUTLASS_DEVICE
  ConstTensorRef reference() const { return ConstTensorRef(first.reference(), second.reference()); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Manages a pair of tile allocations as if they are one allocation
template <typename First_, typename Second_, typename Third_>
struct ZipTileAllocationTriple {
  //
  // Type definitions
  //

  /// First tensor allocation
  typedef First_ First;

  /// Second tensor allocation
  typedef Second_ Second;

  /// meta data tensor allocation
  typedef Third_ Third;

  /// Defines the tensor reference for this allocation
  typedef Zip3TensorRef<typename First::TensorRef, 
                        typename Second::TensorRef,
                        typename Third::TensorRef> TensorRef;

  /// Defines the tensor reference for this allocation
  typedef Zip3TensorRef<typename First::ConstTensorRef, 
                        typename Second::ConstTensorRef,
                        typename Third::ConstTensorRef>
      ConstTensorRef;

  //
  // Data members
  //

  /// First tensor allocation
  First first;

  /// Second tensor allocation
  Second second;

  /// meta data tensor
  Third third;
  //
  // Methods
  //

  /// Returns a TensorRef object pointing to the data
  CUTLASS_DEVICE
  TensorRef reference() { 
    return TensorRef(first.reference(), second.reference(), third.reference()); 
  }

  /// Returns a TensorRef object pointing to the data
  CUTLASS_DEVICE
  ConstTensorRef reference() const { 
    return ConstTensorRef(first.reference(), second.reference(), third.reference()); 
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
