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
    \brief Defines Fragment, a statically-sized array for storing parts of matrices within a
   thread's registers.
*/
#pragma once

#include <assert.h>
#include "cutlass/shape.h"
#include "cutlass/util/cutlass_math.h"
#include "cutlass/vector.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/*!@defgroup fragment_concept Fragment Concept
@{

\ref fragment_concept is a statically sized array for storing parts of tiles held by individual CUDA
threads.

@par \ref fragment_concept
   Types satisfying \ref fragment_concept define the following members
  - <b>Element</b> - type of each access held within the fragment
  - <b>kElements</b> - number of elements stored by the fragment
  - <b>clear()</b> - overwrites the fragment storage with zeros
  - <b>Element & operator[](int i)</b> - by-reference access of the ith element
  - <b>Element const & operator[](int i) const</b> - const by-reference access of the ith element
@}
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

/*!@defgroup fragment_iterator_concept Fragment Iterator Concept
@{

\ref fragment_iterator_concept provides structured access to the elements within a fragment with an
optional bitcast to the desired access type

@par \ref fragment_iterator_concept
   Types satisfying \ref fragment_iterator_concept define the following members
  - <b>AccessType& operator[](int i)</b> - provides access to the ith element of the fragment
  - <b>AccessType& at(int d, int h, int w, int c)</b> - applies \ref layout_concept to fragment and
provides access to element at (d, h, w, c)

@}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int alignment>
struct StorageType {
  typedef uint64_t Type;
};
template <>
struct StorageType<4> {
  typedef uint32_t Type;
};
template <>
struct StorageType<2> {
  typedef uint16_t Type;
};
template <>
struct StorageType<1> {
  typedef uint8_t Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief A template defining \ref fragment_concept
* @concept{fragment_concept}
*/
template <typename Element_, int kElements_, size_t kAlignment_ = 16>
struct Fragment : public AlignedStruct<kAlignment_> {
  /// Make sure the alignment makes sense wrt the size of elements.
  static_assert(int(kAlignment_) == 16 || int(kAlignment_) >= sizeof(Element_), "Alignment is too small");
  /// Alignment must be a power of two
  static_assert(is_pow2<int(kAlignment_)>::value, "Alignment must be a power of two");

  /// This class.
  typedef Fragment<Element_, kElements_> This_;
  /// The element.
  typedef Element_ Element;
  /// The number of elements.
  static int const kElements = kElements_;
  /// Alignment
  static int const kAlignment = int(kAlignment_);

  /// Clear a fragment.
  CUTLASS_HOST_DEVICE void clear() {
    // Avoid element-wise access for sub 32b element type
    if (kAlignment_ >= 8 && (kElements * sizeof(Element)) % 8 == 0) {
      uint64_t* ptr = reinterpret_cast<uint64_t*>(storage);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < (kElements * sizeof(Element)) / 8; ++i) {
        ptr[i] = uint64_t(0);
      }
    } else if (kAlignment_ >= 4 && (kElements * sizeof(Element)) % 4 == 0) {
      uint32_t* ptr = reinterpret_cast<uint32_t*>(storage);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < (kElements * sizeof(Element)) / 4; ++i) {
        ptr[i] = uint32_t(0);
      }
    } else if (kAlignment_ >= 2 && (kElements * sizeof(Element)) % 2 == 0) {
      uint16_t* ptr = reinterpret_cast<uint16_t*>(storage);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < (kElements * sizeof(Element)) / 2; ++i) {
        ptr[i] = uint16_t(0);
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElements; ++i) {
        storage[i] = 0;
      }
    }
  }

  /// The accessor.
  CUTLASS_HOST_DEVICE Element& operator[](int i) { return reinterpret_cast<Element*>(storage)[i]; }

  /// The accessor.
  CUTLASS_HOST_DEVICE Element const& operator[](int i) const {
    return reinterpret_cast<Element const*>(storage)[i];
  }

 private:
  /// Storage type to use for Elements
  typedef typename StorageType<int(kAlignment_)>::Type StorageType;

  /// Number of elements in the storage
  static int const kStorageCount =
      (sizeof(Element_) * kElements_ + sizeof(StorageType) - 1) / sizeof(StorageType);
  /// The storage.
  StorageType storage[kStorageCount];

  /// Ensure that there's enough storage for all elements
  static_assert(sizeof(StorageType) <= kAlignment_, "StorageType is too big for given alignment");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief A template defining \ref fragment_iterator_concept
* @concept{fragment_iterator_concept}
*/
template <typename Fragment_, typename Iterations_, typename AccessType_>
struct FragmentIterator {
  /// This class.
  typedef FragmentIterator<Fragment_, Iterations_, AccessType_> This_;
  /// The fragment.
  typedef Fragment_ Fragment;
  /// The number of iterations.
  typedef Iterations_ Iterations;
  /// The access type.
  typedef AccessType_ AccessType;

  /// The element.
  typedef typename Fragment::Element Element;
  /// The number of elements per access.
  static int const kElementsPerAccess = (int)(sizeof(AccessType) / sizeof(Element));
  /// The shape of the the fragment.
  typedef typename ShapeMul<Iterations, Shape<1, 1, 1, kElementsPerAccess> >::Shape FragmentShape;
  /// The linear strides for iterations.
  typedef typename ShapeStrides<FragmentShape, kElementsPerAccess>::Shape Strides;

  /// Ctor.
  template <typename OtherFragment_>
  CUTLASS_HOST_DEVICE FragmentIterator(OtherFragment_& fragment, int offset = 0)
      : pointer(reinterpret_cast<Element*>(&fragment[offset])) {
    static_assert(OtherFragment_::kElements >= Fragment::kElements, "");
  }

  /// The accessor.
  CUTLASS_HOST_DEVICE AccessType const& at(int d, int h, int w, int c = 0) const {
    int const imm = ComputeOffsetFromStrides<Strides>::get(d, h, w, c);
    return reinterpret_cast<AccessType const&>(pointer[imm]);
  }

  /// The accessor.
  CUTLASS_HOST_DEVICE AccessType& at(int d, int h, int w, int c = 0) {
    int const imm = ComputeOffsetFromStrides<Strides>::get(d, h, w, c);
    return reinterpret_cast<AccessType&>(pointer[imm]);
  }

  /// The accessor.
  CUTLASS_HOST_DEVICE AccessType const& operator[](int i) const {
    return reinterpret_cast<AccessType const&>(pointer[i * kElementsPerAccess]);
  }

  /// The accessor.
  CUTLASS_HOST_DEVICE AccessType& operator[](int i) {
    return reinterpret_cast<AccessType&>(pointer[i * kElementsPerAccess]);
  }

  /// Is the iterator valid?
  CUTLASS_HOST_DEVICE bool valid(int d, int h, int w, int c) const { return true; }

  /// The pointer.
  Element* pointer;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment_, typename Iterations_, typename AccessType_>
struct FragmentConstIterator {
  /// This class.
  typedef FragmentIterator<Fragment_, Iterations_, AccessType_> This_;
  /// The fragment.
  typedef Fragment_ Fragment;
  /// The number of iterations.
  typedef Iterations_ Iterations;
  /// The access type.
  typedef AccessType_ AccessType;

  /// The element.
  typedef typename Fragment::Element Element;
  /// The number of elements per access.
  static int const kElementsPerAccess = (int)(sizeof(AccessType) / sizeof(Element));
  /// The shape of the the fragment.
  typedef typename ShapeMul<Iterations, Shape<1, 1, 1, kElementsPerAccess> >::Shape FragmentShape;
  /// The linear strides for iterations.
  typedef typename ShapeStrides<FragmentShape, kElementsPerAccess>::Shape IterationsStrides;

  /// Ctor.
  template <typename OtherFragment_>
  CUTLASS_HOST_DEVICE FragmentConstIterator(OtherFragment_& fragment, int offset = 0)
      : pointer(reinterpret_cast<Element const*>(&fragment[offset])) {
    static_assert(OtherFragment_::kElements >= Fragment::kElements, "");
  }
  /// Create from non-constant FragmentIterator
  CUTLASS_HOST_DEVICE FragmentConstIterator(
      FragmentIterator<Fragment_, Iterations_, AccessType_> const& rhs_)
      : pointer(reinterpret_cast<Element const*>(rhs_.offset)) {}

  /// The accessor.
  CUTLASS_HOST_DEVICE AccessType const& at(int d, int h, int w, int c = 0) const {
    int const imm = ComputeOffsetFromStrides<IterationsStrides>::get(d, h, w, c);
    return reinterpret_cast<AccessType const&>(pointer[imm]);
  }

  /// The accessor.
  CUTLASS_HOST_DEVICE AccessType const& operator[](int i) const {
    return reinterpret_cast<AccessType const&>(pointer[i * kElementsPerAccess]);
  }

  /// Is the iterator valid?
  CUTLASS_HOST_DEVICE bool valid(int d, int h, int w, int c) const { return true; }

  /// The pointer.
  Element const* pointer;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
