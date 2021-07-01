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
    \brief Introduces TensorRefCollection concept and defines TensorRefBatch and TensorRefArray.
*/

#pragma once

#include "cutlass/tensor_ref.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TensorRefCollection is a concept for storing a logical collection of TensorRef objects. Classes
// satisfying the TensorRefCollection concept must support the following:
//
//   // Define storage type
//   typedef typename TensorRefCollection::Storage Storage;
//
//   // Define a type for offsets in memory
//   typedef typename TensorRefCollection::LongIndex LongIndex;
//
//   // Define a ConstIterator type satisfying TensorRefIterator
//   typedef typename TensorRefCollection::ConstIterator TensorRefIterator;
//
//   // Implement a begin() method.
//   TensorRefIterator iterator = collection.begin();
//
//
// TensorRefIterator is a concept for accessing an element in a TensorRefCollection. Classes
// satisfying the TensorRefIterator concept must support the following:
//
//   // Define a TensorRef type accessed by the iterator
//   typedef typename TensorRefIterator::TensorRef TensorRef;
//
//   // Access the TensorRef
//   TensorRef ref = *iterator;
//
//   // Pre-increment and post-increment
//   ++iterator;
//   iterator++;
//
//   // Pre-decrement and post-decrement
//   --iterator;
//   iterator--;
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// This satisfies TensorRefCollection and stores a collection of TensorRef objects that
/// have identical strides. TensorRef objects are separated by a linear stride.
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
struct TensorRefBatchStrided:
  public TensorRef<Storage_, Rank_, MapFunc_, StorageRank_, Index_, LongIndex_> {

  //
  // Type definitions
  //

  /// Underlying TensorRef type
  typedef TensorRef<Storage_, Rank_, MapFunc_, StorageRank_, Index_, LongIndex_> Base;

  /// Storage type
  typedef typename Base::Storage Storage;

  /// Rank of the logical tensor
  static int const kRank = Rank_;

  /// Index type
  typedef Index_ Index;

  /// Typically, strides in memory can be very large
  typedef LongIndex_ LongIndex;


  /// Coordinate in logical tensor space
  typedef Coord<kRank> TensorCoord;

  /// Tensor reference implied by the TensorRefBatchStrided
  typedef Base TensorRef;

  /// Constant iterator over tensors implied by TensorRefBatchStrided
  class ConstIterator {
  public:
    /// TensorRef returned by the iterator
    typedef Base TensorRef;

  private:

    /// Reference to the parent TensorBatchRef object
    TensorRefBatchStrided const &ref_;

    /// Offset from the base TensorRef pointer
    LongIndex offset_;

  public:

    /// Constructs a ConstIterator from a parent TensorRefBatchStrided
    CUTLASS_HOST_DEVICE
    ConstIterator(
      TensorRefBatchStrided const &ref,
      LongIndex offset = 0): ref_(ref), offset_(offset) { }

    /// Obtains a TensorRef pointed to by the iterator
    CUTLASS_HOST_DEVICE
    TensorRef operator*() const {
      TensorRef ref(ref_);
      ref.add_pointer_offset(offset_);
      return ref;
    }

    /// Advances the iterator to point to the next tensor
    CUTLASS_HOST_DEVICE
    ConstIterator &operator++() {
      offset_ += ref_.tensor_stride;
      return *this;
    }

    /// Advances the iterator to point to the next tensor
    CUTLASS_HOST_DEVICE
    ConstIterator operator++(int) {
      ConstIterator ret(*this);
      offset_ += ref_.tensor_stride;
      return ret;
    }

    /// Returns an iterator advanced by (idx) amount
    CUTLASS_HOST_DEVICE
    ConstIterator operator+(Index idx) {
      return ConstIterator(ref_, offset_ + ref_.tensor_stride * idx);
    }

    /// Advances this iterator by (idx) and returns a reference to self
    CUTLASS_HOST_DEVICE
    ConstIterator &operator+=(Index idx) {
      offset_ += ref_.tensor_stride * idx;
      return *this;
    }

    /// Moves to the previous tensor
    CUTLASS_HOST_DEVICE
    ConstIterator &operator--() {
      offset_ -= ref_.tensor_stride;
      return *this;
    }

    /// Moves to the previous tensor
    CUTLASS_HOST_DEVICE
    ConstIterator operator--(int) {
      ConstIterator ret(*this);
      offset_ -= ref_.tensor_stride;
      return ret;
    }

    /// Returns an iterator moved forward by (idx) amount
    CUTLASS_HOST_DEVICE
    ConstIterator operator-(Index idx) {
      return ConstIterator(ref_, offset_ - ref_.tensor_stride * idx);
    }

    /// Moves this iterator by (idx) and returns a reference to self
    CUTLASS_HOST_DEVICE
    ConstIterator &operator-=(Index idx) {
      offset_ -= ref_.tensor_stride * idx;
      return *this;
    }

    /// Returns the difference in offset between two iterators
    CUTLASS_HOST_DEVICE
    LongIndex operator-(ConstIterator const &it) {
      return offset_ - it.offset_;
    }
  };

  //
  // Data members
  //

  /// Stride between tensors
  LongIndex tensor_stride;

  //
  // Methods
  //

  // Default ctor
  CUTLASS_HOST_DEVICE
  TensorRefBatchStrided(): tensor_stride(0) { }

  // Constructs form a tensor reference and
  CUTLASS_HOST_DEVICE
  TensorRefBatchStrided(TensorRef const &ref, LongIndex _tensor_stride = 0):
    TensorRef(ref),
    tensor_stride(_tensor_stride) { }

  /// Gets the pointer offset
  CUTLASS_HOST_DEVICE
  LongIndex get_pointer_offset(Index idx) const {
    return idx * tensor_stride;
  }

  // Returns a reference
  CUTLASS_HOST_DEVICE
  TensorRef at(Index idx = 0) const {
    TensorRef ref(*this);
    ref.add_pointer_offset(get_pointer_offset(idx));
    return ref;
  }

  /// Returns an iterator
  CUTLASS_HOST_DEVICE
  ConstIterator begin() {
    return ConstIterator(*this);
  }
};

/// Helper to construct a TensorRefBatchStrided<> object using type deduction
template <typename TensorRef_>
CUTLASS_HOST_DEVICE
TensorRefBatchStrided<
  typename TensorRef_::Storage,
  TensorRef_::kRank,
  typename TensorRef_::MapFunc,
  TensorRef_::kStorageGrank,
  typename TensorRef_::Index,
  typename TensorRef_::LongIndex
> make_TensorRefBatchStrided(
  TensorRef_ const &ref,
  typename TensorRef_::LongIndex batch_stride = 0) {

  return TensorRefBatchStrided<
    typename TensorRef_::Storage,
    TensorRef_::kRank,
    typename TensorRef_::MapFunc,
    TensorRef_::kStorageGrank,
    typename TensorRef_::Index,
    typename TensorRef_::LongIndex
  >(ref, batch_stride);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// This satisfies TensorRefCollection and stores a collection of TensorRef objects. This is a
/// structure of arrays in that the individual members of the TensorRef are held in distinct arrays.
///
/// Note, TensorRef maps a logical coordinate space to an n-D array with rank kStorageRank. It
/// maintains a stride vector of similar rank, but the least significant rank is defined to be 1.
///
/// The least significant stride of 1 is not stored, and therefore the number of stride arrays is
/// kStorageRank - 1.
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
struct TensorRefArray {
  //
  // Type definitions
  //

  /// Element pointed to by the TensorRef
  typedef Storage_ Storage;

  /// Index type
  typedef Index_ Index;

  /// Typically, strides in memory can be very large
  typedef LongIndex_ LongIndex;

  /// Rank of the stride vector
  static int const kStorageRank = StorageRank_;

  /// TensorRefIterator over TensorRef objects in TensorRefArray
  class ConstIterator {
  public:

    /// Containing class's tensor rev
    typedef TensorRef<Storage_, Rank_, MapFunc_, StorageRank_, Index_, LongIndex_> TensorRef;

  private:

    /// Reference to the TensorRefArray
    TensorRefArray const &ref_;

    /// Index into TensorRefArray
    int idx_;

  public:

    /// Constructs a ConstIterator over the TensorRef objects
    CUTLASS_HOST_DEVICE
    ConstIterator(TensorRefArray const &ref, int idx = 0): ref_(ref), idx_(idx) { }

    /// Obtains a TensorRef pointed to by this iterator
    CUTLASS_HOST_DEVICE
    TensorRef operator*() const {
      return ref_.reference(idx_);
    }

    /// Advances to next TensorRef
    CUTLASS_HOST_DEVICE
    ConstIterator &operator++() {
      ++idx_;
      return *this;
    }

    /// Advances to next TensorRef
    CUTLASS_HOST_DEVICE
    ConstIterator operator++(int) {
      ConstIterator ret(*this);
      idx_ ++;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    ConstIterator operator+(Index idx) {
      return ConstIterator(ref_, idx_ + idx);
    }

    CUTLASS_HOST_DEVICE
    ConstIterator &operator+=(Index idx) {
      idx_ += idx;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    ConstIterator &operator--() {
      --idx_;
      return *this;
    }

    /// Advances to next TensorRef
    CUTLASS_HOST_DEVICE
    ConstIterator operator--(int) {
      ConstIterator ret(*this);
      --idx_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    ConstIterator &operator-=(Index idx) {
      idx_ -= idx;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    ConstIterator operator-(Index idx) {
      return ConstIterator(ref_, idx_ + idx);
    }
  };

  /// TensorRef type obtained from the TensorRefArray
  typedef TensorRef<Storage_, Rank_, MapFunc_, StorageRank_, Index_, LongIndex_> TensorRef;

  //
  // Data members
  //

  /// Base addresses
  Storage **pointers;

  /// Array of strides
  Index *strides[kStorageRank - 1];

  //
  // Methods
  //

  // Default ctor
  CUTLASS_HOST_DEVICE
  TensorRefArray() { }

  // Construct from pointers to arrays to strides
  CUTLASS_HOST_DEVICE
  TensorRefArray(
    Storage **_pointers,
    Index _strides[kStorageRank - 1]): pointers(_pointers) {

    // Copy pointers to strides arrays
    for (int i = 0; i < kStorageRank - 1; ++i) {
      strides[i] = _strides[i];
    }
  }

  // Returns a TensorRef at the given index in the collection
  CUTLASS_HOST_DEVICE
  TensorRef at(Index idx = 0) const {
    Coord<kStorageRank - 1, Index> stride;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kStorageRank - 1; ++i) {
      stride[i] = strides[idx][i];
    }
    return TensorRef(pointers[idx], stride);
  }

  /// Returns an TesnorRefIterator over the TensorRef objects in this collection
  CUTLASS_HOST_DEVICE
  ConstIterator begin() {
    return ConstIterator(*this);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
