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
    \brief Defines a 1D vector of elements held in the registers of each thread.
*/
#pragma once

#if !defined(__CUDACC_RTC__) || defined(CUTLASS_NVRTC_HAS_FP16)
#include <cuda_fp16.h>
#endif

#include "cutlass/util/numeric_types.h"
#include "cutlass/util/platform.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <size_t kAlignment_>
struct AlignedStruct {};

template <>
struct __align__(1) AlignedStruct<1>{};
template <>
struct __align__(2) AlignedStruct<2>{};
template <>
struct __align__(4) AlignedStruct<4>{};
template <>
struct __align__(8) AlignedStruct<8>{};
template <>
struct __align__(16) AlignedStruct<16>{};
template <>
struct __align__(32) AlignedStruct<32>{};
template <>
struct __align__(64) AlignedStruct<64>{};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kLanes_>
union Vector {
  /// The scalar type.
  typedef Scalar_ Scalar;

  /// The number of elements in the vector.
  enum { kLanes = kLanes_ };
  /// The size of the vector.
  enum { kVectorSize = kLanes * (int)sizeof(Scalar) };
  /// The number of registers needed to store the vector.
  enum { kRegisters = kVectorSize < 4 ? 1 : kVectorSize / 4 };

  // Make sure that the vector type makes sense.
  static_assert(kVectorSize <= 16, "Vector type is too large");

  /// The aligned storage to make sure we have good alignment.
  AlignedStruct<kVectorSize> aligned_;
  /// The associated array of scalars.
  Scalar scalars[kLanes];
  /// The data in registers.
  uint32_t registers[kRegisters];

  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE Scalar const& operator[](uint32_t i) const { return scalars[i]; }
  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE Scalar& operator[](uint32_t i) { return scalars[i]; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__CUDACC_RTC__) || defined(CUTLASS_NVRTC_HAS_FP16)

template <>
union Vector<half, 1> {
  /// The scalar type.
  typedef half Scalar;

  /// The number of elements in the vector.
  enum { kLanes = 1 };
  /// The size of the vector.
  enum { kVectorSize = kLanes * (int)sizeof(Scalar) };
  /// The number of registers needed to store the vector.
  enum { kRegisters = kVectorSize < 4 ? 1 : kVectorSize / 4 };

  // Make sure that the vector type makes sense.
  static_assert(kVectorSize <= 16, "Vector type is too large");

  /// The aligned storage to make sure we have good alignment.
  AlignedStruct<kVectorSize> aligned_;
  /// The associated array of scalars.
  uint16_t scalars[kLanes];

  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE Scalar const& operator[](uint32_t i) const {
    return reinterpret_cast<Scalar const&>(scalars[i]);
  }
  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE Scalar& operator[](uint32_t i) {
      return reinterpret_cast<Scalar&>(scalars[i]);
  }
};


template <int kLanes_>
union Vector<half, kLanes_> {
  /// The scalar type.
  typedef half Scalar;

  /// The number of elements in the vector.
  enum { kLanes = kLanes_ };
  /// The size of the vector.
  enum { kVectorSize = kLanes * (int)sizeof(Scalar) };
  /// The number of registers needed to store the vector.
  enum { kRegisters = kVectorSize < 4 ? 1 : kVectorSize / 4 };

  // Make sure that the vector type makes sense.
  static_assert(kVectorSize <= size_t(16), "Vector type is too large");

  /// The aligned storage to make sure we have good alignment.
  AlignedStruct<kVectorSize> aligned_;
  /// The associated array of scalars.
  uint16_t scalars[kLanes];
  /// The data in registers.
  uint32_t registers[kRegisters];

  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE Scalar const& operator[](uint32_t i) const {
    return reinterpret_cast<Scalar const&>(scalars[i]);
  }
  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE Scalar& operator[](uint32_t i) {
      return reinterpret_cast<Scalar&>(scalars[i]);
  }
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Vector definition for 1-bit binary datatype
template <int kLanes_>
union Vector<bin1_t, kLanes_> {
  /// The scalar type.
  typedef bin1_t Scalar;

  /// The number of elements in the vector.
  enum { kLanes = kLanes_ };
  /// The size of the vector.
  enum { kVectorSize = kLanes / 8 };
  /// The number of registers needed to store the vector.
  enum { kRegisters = kVectorSize < 4 ? 1 : kVectorSize / 4 };

  static_assert((kLanes >= 8) && !(kLanes % 8),
                "May only construct vectors of bin1_t that are multiples of 8 bits.");

  /// The aligned storage to make sure we have good alignment.
  AlignedStruct<kVectorSize> aligned_;
  /// The data in registers.
  uint32_t registers[kRegisters];

  /// Default Constructor
  CUTLASS_HOST_DEVICE
  Vector() {}
  /// Constructor to convert from uint32_t type
  CUTLASS_HOST_DEVICE Vector(uint32_t value) { registers[0] = value; }
  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE bool operator[](uint32_t i) const {
    return ( (registers[i / 32] & (1 << (i % 32))) != 0 );
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Vector definition for 4-bit signed integer datatype
template <int kLanes_>
union Vector<int4_t, kLanes_> {
  /// The scalar type.
  typedef int4_t Scalar;

  /// The number of elements in the vector.
  enum { kLanes = kLanes_ };
  /// The size of the vector.
  enum { kVectorSize = kLanes / 2 };
  /// The number of registers needed to store the vector.
  enum { kRegisters = kVectorSize < 4 ? 1 : kVectorSize / 4 };

  static_assert((kLanes >= 2) && !(kLanes % 2),
   "May only construct vectors of int4_t that are multiples of 8 bits.");

  /// The aligned storage to make sure we have good alignment.
  AlignedStruct<kVectorSize> aligned_;
  /// The data in registers.
  uint32_t registers[kRegisters];

  /// Default Constructor
  CUTLASS_HOST_DEVICE
  Vector() {}
  /// Constructor to convert from uint32_t type
  CUTLASS_HOST_DEVICE Vector(uint32_t value) { registers[0] = value; }
  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE int operator[](uint32_t i) const {
    return (registers[i / 8] >> (i % 8 * 4) & 0x0f)
              - 16 * (registers[i / 8] >> (i % 8 * 4 + 3) & 0x01);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Vector definition for 4-bit unsigned integer datatype
template <int kLanes_>
union Vector<uint4_t, kLanes_> {
  /// The scalar type.
  typedef uint4_t Scalar;

  /// The number of elements in the vector.
  enum { kLanes = kLanes_ };
  /// The size of the vector.
  enum { kVectorSize = kLanes / 2 };
  /// The number of registers needed to store the vector.
  enum { kRegisters = kVectorSize < 4 ? 1 : kVectorSize / 4 };

  static_assert((kLanes >= 2) && !(kLanes % 2),
    "May only construct vectors of uint4_t that are multiples of 8 bits.");

  /// The aligned storage to make sure we have good alignment.
  AlignedStruct<kVectorSize> aligned_;
  /// The data in registers.
  uint32_t registers[kRegisters];

  /// Default Constructor
  CUTLASS_HOST_DEVICE
  Vector() {}
  /// Constructor to convert from uint32_t type
  CUTLASS_HOST_DEVICE Vector(uint32_t value) { registers[0] = value; }
  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE int operator[](uint32_t i) const {
    return registers[i / 8] >> (i % 8 * 4) & 0x0f;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Vector definition for 4-bit signed integer datatype
template <int kLanes_>
union Vector<int8_t, kLanes_> {
  /// The scalar type.
  typedef int8_t Scalar;

  /// The number of elements in the vector.
  enum { kLanes = kLanes_ };
  /// The size of the vector.
  enum { kVectorSize = kLanes };
  /// The number of registers needed to store the vector.
  enum { kRegisters = kVectorSize < 4 ? 1 : (kVectorSize+3) / 4 };

//  static_assert((kLanes >= 2) && !(kLanes % 2),
//   "May only construct vectors of int8_t that are multiples of 8 bits.");

  /// The aligned storage to make sure we have good alignment.
  AlignedStruct<kVectorSize> aligned_;
  /// The data in registers.
  uint32_t registers[kRegisters];

  /// Default Constructor
  CUTLASS_HOST_DEVICE
  Vector() {}
  /// Constructor to convert from uint32_t type
  CUTLASS_HOST_DEVICE Vector(uint32_t value) { registers[0] = value; }
  /// Accessor to the ith lane.
  CUTLASS_HOST_DEVICE int operator[](uint32_t i) const {
    return (registers[i / 4] >> (i % 4 * 8) & 0xff);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_>
CUTLASS_HOST_DEVICE void make_zero(Scalar_& x) {
  x = Scalar_(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element_, int kLanes_ = 1>
struct Vectorize {
  typedef Vector<Element_, kLanes_> Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kLanes_>
struct Vectorize<Vector<bin1_t, 32>, kLanes_> {
  typedef Vector<bin1_t, kLanes_ * 32> Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kLanes_>
struct Vectorize<Vector<int4_t, 8>, kLanes_> {
  typedef Vector<int4_t, kLanes_ * 8> Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kLanes_>
struct Vectorize<Vector<uint4_t, 8>, kLanes_> {
  typedef Vector<uint4_t, kLanes_ * 8> Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kLanes_>
CUTLASS_HOST_DEVICE void make_zero(Vector<Scalar_, kLanes_>& vec) {
  for (int i = 0; i < Vector<Scalar_, kLanes_>::kRegisters; ++i) {
    vec.registers[i] = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// cutlass::Extent similar to std::extent but applicable to CUTLASS types
//

/// Returns the extent of a scalar or vector
template <typename T>
struct Extent {
  static size_t const kValue = 1;
};

/// Returns the number of lanes of a vector if need be
template <typename T, int Lanes>
struct Extent<Vector<T, Lanes> > {
  static size_t const kValue = Lanes;
};

/// Returns the number of lanes of a vector if need be
template <typename T, int Lanes>
struct Extent<Vector<T, Lanes> const> {
  static size_t const kValue = Lanes;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Traits describing properties of vectors and scalar-as-vectors
template <typename T>
struct VectorTraits {
  /// Scalar type
  typedef T Scalar;

  /// Number of lanes of vector
  static int const kLanes = 1;

  /// True if the type is actually a cutlass::Vector, otherwise false
  static bool const IsVector = false;

  /// Type that is always a vector
  typedef Vector<T, 1> Vector;
};

/// Partial specialization for actual cutlass::Vector
template <typename T, int Lanes>
struct VectorTraits<Vector<T, Lanes> > {
  /// Scalar type
  typedef T Scalar;

  /// Number of lanes of vector
  static int const kLanes = Lanes;

  /// Type is actually a cutlass::Vector
  static bool const IsVector = true;

  /// Type that is always a Vector
  typedef Vector<T, Lanes> Vector;
};

/// Partial specialization for actual cutlass::Vector
template <typename T, int Lanes>
struct VectorTraits<Vector<T, Lanes> const> {
  /// Scalar type
  typedef T Scalar;

  /// Number of lanes of vector
  static int const kLanes = Lanes;

  /// Type is actually a cutlass::Vector
  static bool const IsVector = true;

  /// Type that is always a Vector
  typedef Vector<T, Lanes> Vector;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
