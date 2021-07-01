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
    \brief Defines a structure containing strides, bounds, and a pointer to tensor data.
*/
#pragma once

#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/vector.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Default mapping function from coordinates in a tensor's index space into the n-D array held
/// in memory. Assumes StorageRank = Rank
template <int Rank>
struct IdentityTensorMapFunc {
  static int const kStorageRank = Rank;
  CUTLASS_HOST_DEVICE
  Coord<Rank> operator()(Coord<Rank> const &coord) const {
    return coord;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/* \brief Structure modeling a pointer and stride into a tensor.

  A tensor consists of an index space with Rank_ dimensions. It is stored in memory modeled
  as an n-D array, where n = StorageRank_. A mapping function maps the logical coordinates of the
  tensor's index space into the n-D array, and a stride vector maps the n-D array to linear memory.

  CUTLASS requires the n-D array's least significant, "fastest changing" dimension to
  be contiguous in memory. It therefore has a stride of 1 and is not stored. Construction is offered
  from vectors of full StorageRank and of the 'compact' rank, though it is in error to construct
  with the least significant stride != 1.

  The requirement that the least significant dimension be consecutive enables numerous optimizations
  and assumptions about vectorizing memory accesses throughout CUTLASS. It also matches various
  BLAS conventions in which only the "leading dimension" or most significant stride of a rank=2
  matrix is provided.
  
  Examples:

  (These examples use helpers for matrix layouts defined in cutlass/matrix_traits.h)

  1. Column-major matrix may be represented as a rank=2 tensor:

    TensorRef<float, 2, MatrixLayout::ColumnMajor> A(ptr_A, make_Coord(ldm, 1));

  2. Row-major matrix may be represented as a rank=2 tensor:

    TensorRef<float, 2, MatrixLayout::RowMajor> B(ptr_A, ldm);

  3. An interleaved matrix may be represented as a rank=2 tensor:

    TensorRef<int8_t, 2, MatrixLayout::ColumnMajorInterleaved<32> > C;

  4. Defining a matrix with arbitrary strides in each dimension

    struct ContiguousLayout {

      /// Arbitrary storage rank
      static int const kStorageRank = 3;

      /// Mapping function defined by runtime stride configuration
      CUTLASS_HOST_DEVICE
      Coord<3> operator()(MatrixCoord const &coord) const {
          return make_Coord(coord.row(), coord.column(), 0);
      }
    };

    typedef TensorRef<float, 2, ContiguousLayout> ContiguousTensorRef;

    // Construct the TensorRef object from a pair of stride values
    ContiguousTensorRef D(ptr_D, make_Coord(row_stride, column_stride));


  5. A helper exists to define a TensorRef for a contiguous matrix whose layout
     is not known at compile time.

    MatrixLayout::Kind layout;   // Could be MatrixLayout::kRowMajor or MatrixLayout::kColumnMajor
    int ldm;                     // leading dimension

    ContiguousTensorRef E(ptr_E, ContiguousLayout::stride(layout, ldm));

*/
template <
  /// Data type of element stored within tensor
  typename Storage_,
  /// Rank of logical tensor
  int Rank_,
  /// Maps a Coord<Rank_> in the logical tensor index space to the internal n-D array
  typename MapFunc_ = IdentityTensorMapFunc<Rank_>,
  /// Rank of internal n-D array
  int StorageRank_ = MapFunc_::kStorageRank,
  /// Index type used for coordinates
  typename Index_ = int,
  /// Index type used for offsets and pointer differences
  typename LongIndex_ = long long
>
class TensorRef {
 public:
  /// Data type of individual access
  typedef Storage_ Storage;

  /// Logical rank of tensor index space
  static int const kRank = Rank_;

  /// Mapping function from logical coordinate to internal n-D array
  typedef MapFunc_ MapFunc;

  /// Rank of internal storage
  static int const kStorageRank = StorageRank_;

  /// Index type
  typedef Index_ Index;

  /// Typically, strides in memory can be very large
  typedef LongIndex_ LongIndex;

  /// Coordinate in logical tensor space
  typedef Coord<kRank> TensorCoord;

  /// Coordinate in storage n-D array
  typedef Coord<kStorageRank> StorageCoord;

  /// Stride vector in storage coordinage space - assumes least significant stride
  /// is 1 and does not store it.
  typedef Coord<kStorageRank - 1> StrideVector;

  /// Tensor reference to of constant value
  typedef TensorRef<
    typename platform::remove_const<Storage>::type const,
    Rank_,
    MapFunc_,
    StorageRank_,
    Index_,
    LongIndex_> ConstTensorRef;

  /// Require at least rank=1. Mathematically, a rank=0 tensor would be considered to be a
  /// scalar, but degenerate cases such as these are difficult to accommodate without
  /// extensive C++ metaprogramming or support for zero-length arrays.
  static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

  //
  // Definitions included for backwards compatibility - to be removed in next major release
  //

  /// Coordinate in logical tensor space
  typedef TensorCoord Coord_t;

  /// Logical rank of tensor index space
  static int const Rank = kRank;

 private:

  /// Pointer
  Storage* ptr_;

  /// Stride vector - fastest-changing stride assumed to be 1 and not stored
  StrideVector stride_;

  /// Maps a logical coordinate to an n-D array's tensor space
  MapFunc coord_map_;

 public:

  //
  // Methods
  //

  /// Helper for 1-D memory. All higher ranks are projected onto the fastest changing rank.
  CUTLASS_HOST_DEVICE
  TensorRef(Storage *ptr = nullptr): ptr_(ptr) {
    for (int i = 0; i < kStorageRank - 1; ++i) {
      stride_[i] = 1;
    }
  }

  /// Helper to construct from a pointer and single stride element for 2-D pitch linear memory.
  // Higher ranks are projected onto the fastest-changing rank.
  CUTLASS_HOST_DEVICE
  TensorRef(Storage* ptr, Index ldm) {
    ptr_ = ptr;
    for (int i = 0; i < kStorageRank - 1; ++i) {
      stride_[i] = ldm;
    }
  }

  /// Constructs from a single pointer and stride vector
  CUTLASS_HOST_DEVICE
  TensorRef(Storage* ptr, StrideVector const& stride) : ptr_(ptr), stride_(stride) {

  }

  /// Constructs from a pointer and a stride vector of size kRank. If fastest changing
  /// stride is not 1, construction fails and subsequent calls to good() will return false.
  CUTLASS_HOST_DEVICE
  TensorRef(Storage* ptr, StorageCoord const& stride) {
    // Fastest-changing stride must be one
    if (stride.at(kStorageRank - 1) == 1) {
      ptr_ = ptr;
      for (int i = 0; i < kStorageRank - 1; ++i) {
        stride_[i] = stride[i];
      }
    }
    else {
      // Fastest-chaning stride must be 1.
      reset();
    }
  }

  /// Enables conversion from TensorRef of non-const type
  CUTLASS_HOST_DEVICE
  TensorRef(
    TensorRef<
      typename platform::remove_const<Storage>::type,
      kRank,
      MapFunc,
      kStorageRank,
      Index,
      LongIndex> const &ref
  ):
    ptr_(ref.data()) {
    for (int i = 0; i < kStorageRank - 1; ++i) {
      stride_[i] = ref.stride(i);
    }
  }

  /// Returns a reference to constant-valued tensor
  CUTLASS_HOST_DEVICE
  ConstTensorRef const_ref() const {
    return ConstTensorRef(*this);
  }

  /// Updates only the pointer
  CUTLASS_HOST_DEVICE
  void reset(Storage* ptr = nullptr) {
    ptr_ = ptr;
  }

  /// Updates the pointer, stride, and location within a TensorRef
  CUTLASS_HOST_DEVICE
  void reset(Storage* ptr, StorageCoord const & stride) {
    // Fastest-changing stride must be one
    if (stride.at(kStorageRank - 1) == 1) {
      ptr_ = ptr;
      for (int i = 0; i < kStorageRank - 1; ++i) {
        stride_[i] = stride[i];
      }
    }
    else {
      // Fastest-changing stride must be 1 - this is an error.
      reset();
    }
  }

  /// Returns true if the TensorRef may be safely accessed
  CUTLASS_HOST_DEVICE
  bool good() const {
    return ptr_ != nullptr;
  }

  /// Returns the pointer to referenced data
  CUTLASS_HOST_DEVICE
  Storage * data() const { return ptr_; }

  /// Returns the stride of the tensor
  CUTLASS_HOST_DEVICE
  StorageCoord stride() const {
    StorageCoord ld;
    for (int i = 0; i < kStorageRank - 1; ++i) {
      ld[i] = stride_[i];
    }
    ld[kStorageRank - 1] = 1;
    return ld;
  }

  /// Returns the stride of the tensor in the given dimension
  CUTLASS_HOST_DEVICE
  Index stride(int dim) const {
    // fastest-changing stride assumbed to be 1
    if (dim + 1 >= kStorageRank) {
      return 1;
    }
    return stride_.at(dim);
  }

  /// Returns the maximum stride element as the 'leading dimension'
  CUTLASS_HOST_DEVICE
  Index leading_dim(int idx = 0) const { return stride(idx); }

  /// Maps a logical coordinate to an n-D array in memory
  CUTLASS_HOST_DEVICE
  StorageCoord map(TensorCoord const &coord) const {
    return coord_map_(coord);
  }

  /// Computes the offset of an index from the origin of the tensor
  CUTLASS_HOST_DEVICE
  LongIndex offset(TensorCoord const& coord) const {
    return stride().template dot<LongIndex>(map(coord));
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Storage& at(TensorCoord const& coord) const {
    return ptr_[offset(coord)];
  }

  /// Returns a reference to the element at a given linear index
  CUTLASS_HOST_DEVICE
  Storage& at(LongIndex idx) const { return ptr_[idx]; }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Storage& operator[](TensorCoord const& coord) const {
    return ptr_[offset(coord)];
  }

  /// Returns a reference to the element at a given linear index
  CUTLASS_HOST_DEVICE
  Storage& operator[](LongIndex idx) const { return ptr_[idx]; }

  /// Adds an offset to each pointer
  CUTLASS_HOST_DEVICE
  TensorRef & add_pointer_offset(LongIndex delta) {
    ptr_ += delta;
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef operator+(TensorCoord const& b) const {
    TensorRef result(*this);
    result.add_pointer_offset(offset(b));
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef& operator+=(TensorCoord const& b) {
    add_pointer_offset(offset(b));
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef operator-(TensorCoord const& b) const {
    TensorRef result(*this);
    result.add_pointer_offset(-offset(b));
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef& operator-=(TensorCoord const& b) {
    add_pointer_offset(-offset(b));
    return *this;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations to handle degenerate cases.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for rank=1 case with no internal StrideVector
template <
  /// Data type of element stored within tensor
  typename Storage_,
  /// Rank of logical tensor
  int Rank_,
  /// Maps a Coord<Rank_> in the logical tensor index space to the internal n-D array
  typename MapFunc_,
  /// Index type used for coordinates
  typename Index_,
  /// Index type used for offsets and pointer differences
  typename LongIndex_
>
class TensorRef<Storage_, Rank_, MapFunc_, 1, Index_, LongIndex_> {
 public:
  /// Data type of individual access
  typedef Storage_ Storage;

  /// Logical rank of tensor index space
  static int const kRank = Rank_;

  /// Mapping function from logical coordinate to internal n-D array
  typedef MapFunc_ MapFunc;

  /// Rank of internal storage
  static int const kStorageRank = 1;

  /// Index type
  typedef Index_ Index;

  /// Typically, strides in memory can be very large
  typedef LongIndex_ LongIndex;

  /// Coordinate in logical tensor space
  typedef Coord<kRank> TensorCoord;

  /// Coordinate in storage n-D array
  typedef Coord<kStorageRank> StorageCoord;

  /// Stride vector in storage coordinage space - assumes least significant stride
  /// is 1 and does not store it.
  struct StrideVector { };

  /// Tensor reference to of constant value
  typedef TensorRef<
    typename platform::remove_const<Storage>::type const,
    Rank_,
    MapFunc_,
    kStorageRank,
    Index_,
    LongIndex_> ConstTensorRef;

  //
  // Definitions included for backwards compatibility - to be removed in next major release
  //

  /// Coordinate in logical tensor space
  typedef TensorCoord Coord_t;

  /// Logical rank of tensor index space
  static int const Rank = kRank;

 private:

  /// Pointer
  Storage* ptr_;

  /// Maps a logical coordinate to an n-D array's tensor space
  MapFunc coord_map_;

 public:

  //
  // Methods
  //

  /// Helper for 1-D memory. All higher ranks are projected onto the fastest changing rank.
  CUTLASS_HOST_DEVICE
  TensorRef(Storage *ptr = nullptr): ptr_(ptr) { }

  /// Constructs from a single pointer and stride vector
  CUTLASS_HOST_DEVICE
  TensorRef(Storage* ptr, StrideVector const& stride) : ptr_(ptr) {

  }

  /// Constructs from a pointer and a stride vector of size kRank. If fastest changing
  /// stride is not 1, construction fails and subsequent calls to good() will return false.
  CUTLASS_HOST_DEVICE
  TensorRef(Storage* ptr, StorageCoord const& stride) {
    // Fastest-changing stride must be one
    if (stride.at(kStorageRank - 1) == 1) {
      ptr_ = ptr;
    }
    else {
      // Fastest-chaning stride must be 1.
      reset();
    }
  }

  /// Enables conversion from TensorRef of non-const type
  CUTLASS_HOST_DEVICE
  TensorRef(
    TensorRef<
      typename platform::remove_const<Storage>::type,
      kRank,
      MapFunc,
      kStorageRank,
      Index,
      LongIndex> const &ref
  ):
    ptr_(ref.data()) {
  }

  /// Returns a reference to constant-valued tensor
  CUTLASS_HOST_DEVICE
  ConstTensorRef const_ref() const {
    return ConstTensorRef(*this);
  }

  /// Updates only the pointer
  CUTLASS_HOST_DEVICE
  void reset(Storage* ptr = nullptr) {
    ptr_ = ptr;
  }

  /// Updates the pointer, stride, and location within a TensorRef
  CUTLASS_HOST_DEVICE
  void reset(Storage* ptr, StorageCoord const & stride) {
    // Fastest-changing stride must be one
    if (stride.at(kStorageRank - 1) == 1) {
      ptr_ = ptr;
    }
    else {
      // Fastest-changing stride must be 1 - this is an error.
      reset();
    }
  }

  /// Returns true if the TensorRef may be safely accessed
  CUTLASS_HOST_DEVICE
  bool good() const {
    return ptr_ != nullptr;
  }

  /// Returns the pointer to referenced data
  CUTLASS_HOST_DEVICE
  Storage * data() const { return ptr_; }

  /// Returns the pointer to referenced data at the given coordinate
  CUTLASS_HOST_DEVICE
  Storage * data(TensorCoord const& coord) const { return ptr_ + offset(coord); }

  /// Returns the stride of the tensor
  CUTLASS_HOST_DEVICE
  StorageCoord stride() const {
    StorageCoord ld;
    ld[kStorageRank - 1] = 1;
    return ld;
  }

  /// Returns the stride of the tensor in the given dimension
  CUTLASS_HOST_DEVICE
  Index stride(int dim) const {
    // fastest-changing stride assumbed to be 1
    return 1;
  }

  /// Returns the maximum stride element as the 'leading dimension'
  CUTLASS_HOST_DEVICE
  Index leading_dim(int idx = 0) const { return 1; }

  /// Maps a logical coordinate to an n-D array in memory
  CUTLASS_HOST_DEVICE
  StorageCoord map(TensorCoord const &coord) const {
    return coord_map_(coord);
  }

  /// Computes the offset of an index from the origin of the tensor
  CUTLASS_HOST_DEVICE
  LongIndex offset(TensorCoord const& coord) const {
    return stride().template dot<LongIndex>(map(coord));
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Storage& at(TensorCoord const& coord) const {
    return ptr_[offset(coord)];
  }

  /// Returns a reference to the element at a given linear index
  CUTLASS_HOST_DEVICE
  Storage& at(LongIndex idx) const { return ptr_[idx]; }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Storage& operator[](TensorCoord const& coord) const {
    return ptr_[offset(coord)];
  }

  /// Returns a reference to the element at a given linear index
  CUTLASS_HOST_DEVICE
  Storage& operator[](LongIndex idx) const { return ptr_[idx]; }

  /// Adds an offset to each pointer
  CUTLASS_HOST_DEVICE
  TensorRef & add_pointer_offset(LongIndex delta) {
    ptr_ += delta;
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef operator+(TensorCoord const& b) const {
    TensorRef result(*this);
    result.add_pointer_offset(offset(b));
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef& operator+=(TensorCoord const& b) {
    add_pointer_offset(offset(b));
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef operator-(TensorCoord const& b) const {
    TensorRef result(*this);
    result.add_pointer_offset(-offset(b));
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef& operator-=(TensorCoord const& b) {
    add_pointer_offset(-offset(b));
    return *this;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
