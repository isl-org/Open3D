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
    \brief Defines a structure containing strides and a pointer to tensor data.

    TensorView is derived from TensorRef and contributes bounds to the tensor's index space. Thus,
    it is a complete mathematical object and may be used in tensor algorithms. It is decoupled from
    data storage and is therefore lightweight and may be embedded in larger tensor objects or
    memory structures.

    See cutlass/tensor_ref.h for more details about the mapping of the logical tensor index space to
    linear memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a view into a logical tensor
template <
  /// Data type of element stored within tensor
  typename Storage_,
  /// Rank of logical tensor
  int Rank_ = 4,
  /// Maps a Coord<Rank_> in the logical tensor index space to the internal n-D array
  typename MapFunc_ = IdentityTensorMapFunc<Rank_>,
  /// Rank of internal n-D array
  int StorageRank_ = MapFunc_::kStorageRank,
  /// Index type used for coordinates
  typename Index_ = int,
  /// Index type used for offsets and pointer differences
  typename LongIndex_ = long long
>
class TensorView : public TensorRef<Storage_, Rank_, MapFunc_, StorageRank_, Index_, LongIndex_> {
 public:
  /// Base tensor reference
  typedef TensorRef<Storage_, Rank_, MapFunc_, StorageRank_, Index_, LongIndex_> Base;

  /// Tensor reference to of constant value
  typedef TensorRef<
    typename platform::remove_const<Storage_>::type const,
    Rank_,
    MapFunc_,
    StorageRank_,
    Index_,
    LongIndex_> ConstTensorRef;

  /// Base tensor reference
  typedef Base TensorRef_t;

  /// Storage type
  typedef typename Base::Storage Storage;

  /// Index type
  typedef typename Base::Index Index;

  /// Coordinate in logical tensor space
  typedef typename TensorRef_t::TensorCoord TensorCoord;

  /// Coordinate in storage n-D array
  typedef typename TensorRef_t::StorageCoord StorageCoord;

  /// Stride vector in storage coordinate space
  /// Least significant stride is = 1 and not stored
  typedef typename TensorRef_t::StrideVector StrideVector;

  /// TensorView of constant value
  typedef TensorView<
    typename platform::remove_const<Storage>::type const,
    Rank_,
    MapFunc_,
    StorageRank_,
    Index_,
    LongIndex_> ConstTensorView;

  //
  // Definitions included for backwards compatibility - to be removed in next major release
  //

  /// Coordinate in logical tensor space
  typedef TensorCoord Coord_t;

  /// Logical rank of tensor index space
  static int const Rank = Base::kRank;

  /// Type used to compute the offset of an element to the base of a tensor
  typedef typename Base::LongIndex Offset_t;

  /// TensorRef to const-valued type
  typedef typename TensorRef_t::ConstTensorRef ConstTensorRef_t;

 private:
  //
  // Data members
  //

  /// Dimensions of coordinate (independent of stride)
  TensorCoord size_;

 public:
  //
  // Device and Host Methods
  //

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TensorView() {}

  /// Constructs a TensorView from a TensorRef and size
  CUTLASS_HOST_DEVICE
  TensorView(Base const& _ref, TensorCoord const& _size) : Base(_ref), size_(_size) {}

  /// Constructs a TensorView from a pointer, a stride vector, and size
  CUTLASS_HOST_DEVICE
  TensorView(
    Storage *ptr,
    StrideVector const &stride,
    TensorCoord const& size
  ):
    Base(ptr, stride), size_(size) {}

  /// Constructs a TensorView from a pointer, a stride vector, and size
  CUTLASS_HOST_DEVICE
  TensorView(
    Storage *ptr,
    StorageCoord const &stride,
    TensorCoord const& size
  ):
    Base(ptr, stride), size_(size) {}

  /// Updates the reference and size of a Tensor_view object
  CUTLASS_HOST_DEVICE
  void reset(Base const& _ref = Base(), TensorCoord const& _size = TensorCoord()) {
    Base::operator=(_ref);
    size_ = _size;
  }

  /// Accesses the size
  CUTLASS_HOST_DEVICE
  TensorCoord const& size() const { return size_; }

  /// Accesses the size
  CUTLASS_HOST_DEVICE
  Index size(int dim) const { return size_.at(dim); }

  /// Assigns the Tensor_view
  CUTLASS_HOST_DEVICE
  TensorView& operator=(TensorView const& _tensor) {
    Base::operator=(_tensor);
    size_ = _tensor.size_;
    return *this;
  }

  /// Determines whether a location is within a tensor
  CUTLASS_HOST_DEVICE
  bool contains(TensorCoord const& coord) const {
    CUTLASS_PRAGMA_UNROLL
    for (int dim = 0; dim < Rank_; ++dim) {
      if (coord[dim] >= size_[dim]) {
        return false;
      }
    }
    return true;
  }

  /// Determines the order of dims of the tensor (e.g., CHW versus HWC)
  CUTLASS_HOST_DEVICE
  void getStrideOrder(int order[]) const {
    for (int i = 0; i < Rank_; i++) order[i] = i;
    // Bubble sort
    for (int start = 0; start < Rank_ - 1; start++) {
      for (int i = start; i < Rank_ - 1; i++) {
        if (this->stride(order[i]) < this->stride(order[i + 1])) {
          int temp = order[i];
          order[i] = order[i + 1];
          order[i + 1] = temp;
        }
      }
    }
    // post-condition: this->stride(ord[i]) >= this->stride(ord[i+1]) for i from [0,Rank_-2]
  }

  /// Determines if the values in the tensor are contiguous
  CUTLASS_HOST_DEVICE
  bool isPacked() const {
    if (Rank_ <= 0) return true;
    int ord[Rank_];
    getStrideOrder(ord);
    // first check if the slowest dimension has a stride of 1
    if (this->stride(ord[Rank_ - 1]) != 1) return false;
      // now check that there are no gaps between strides
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Rank_; i++)
        if (this->stride(ord[i]) != this->stride(ord[i + 1]) * size_[ord[i + 1]]) return false;
    return true;
  }

  /// Returns a TensorRef pointing to the first element of the tensor.
  CUTLASS_HOST_DEVICE
  TensorRef_t ref() const {
    return TensorRef_t(*this);
  }

  /// Returns a TensorRef_t pointing to the first element of the tensor.
  CUTLASS_HOST_DEVICE
  ConstTensorRef const_ref() const {
    return ConstTensorRef(*this);
  }

  /// Returns a Tensor_view given location and size quantities
  CUTLASS_HOST_DEVICE
  TensorView subview(TensorCoord const& location, TensorCoord size) const {
    return TensorView((*this) + location, size.clamp(size_ - location));
  }

  /// Returns the number of scalar elements needed to store tensor
  CUTLASS_HOST_DEVICE
  size_t capacity() const {
    int max_rank = 0;

    StorageCoord mapped_size(this->map(size()));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Base::kStorageRank; ++i) {
      if (!i ||
        this->stride(i) * mapped_size[i] > this->stride(max_rank) * mapped_size[max_rank]) {
        max_rank = i;
      }
    }
    return this->stride(max_rank) * mapped_size[max_rank];
  }

  /// Returns a TensorView offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorView operator+(TensorCoord const& b) const {
    TensorView result(*this);
    result.add_pointer_offset(this->offset(b));
    return result;
  }

  /// Returns a TensorRef_t offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorView& operator+=(TensorCoord const& b) {
    this->add_pointer_offset(this->offset(b));
    return *this;
  }

  /// Returns a TensorRef_t offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorView operator-(TensorCoord const& b) const {
    TensorRef_t result(*this);
    result.add_pointer_offset(-this->offset(b));
    return result;
  }

  /// Returns a TensorRef_t offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorView& operator-=(TensorCoord const& b) {
    this->add_pointer_offset(-this->offset(b));
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
