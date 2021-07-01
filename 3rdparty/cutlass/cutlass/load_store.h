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
    \brief Defines abstractions for efficiently loading and storing vectors to memory.
*/
#pragma once

#include "cutlass/vector.h"
namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Enum to specify which memory space data resides in.
*/
struct MemorySpace {
  enum Kind {
    kGeneric,  // Data accessed through pointer dereferencing
    kShared,   // Data resides in shared memory
    kGlobal    // Data resides in global memory
  };
};

/// Specifies whether iterator storage fragment consists of Scalar values or WMMA matrix
struct FragmentElementType {
  enum Kind { kScalar, kWmmaMatrix };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          int kAccessSize,
          MemorySpace::Kind Memory_,
          FragmentElementType::Kind kFragmentElementType = FragmentElementType::kScalar,
          typename FragmentElement_ = Scalar_,
          int kStride = 1,
          size_t size = (sizeof(Scalar_) * kAccessSize)>
struct Load {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    dst = *reinterpret_cast<AccessType const*>(pointer + offset);
  }

  /// The clear function.
  static CUTLASS_HOST_DEVICE void clear(AccessType& dst) {
    dst = 0;
  }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for 16b loads
template <typename Scalar_, int kAccessSize, MemorySpace::Kind Memory_>
struct Load<Scalar_, kAccessSize, Memory_, FragmentElementType::kScalar, Scalar_, 1, 2> {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    reinterpret_cast<uint16_t&>(dst) = reinterpret_cast<uint16_t const*>(&pointer[offset])[0];
  }

  /// The clear function.
  static CUTLASS_HOST_DEVICE void clear(AccessType& dst) {
    dst = uint16_t(0);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kAccessSize, MemorySpace::Kind Memory_, int kStride>
struct Load<Scalar_, kAccessSize, Memory_, FragmentElementType::kScalar, Scalar_, kStride, 4> {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    dst.registers[0] = reinterpret_cast<uint32_t const*>(&pointer[offset])[0];
  }

  /// The clear function.
  static CUTLASS_HOST_DEVICE void clear(AccessType& dst) {
    dst.registers[0] = uint32_t(0);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kAccessSize, MemorySpace::Kind Memory_, int kStride>
struct Load<Scalar_, kAccessSize, Memory_, FragmentElementType::kScalar, Scalar_, kStride, 8> {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    uint2 tmp = reinterpret_cast<uint2 const*>(&pointer[offset])[0];
    dst.registers[0] = tmp.x;
    dst.registers[1] = tmp.y;
  }

  /// The clear function.
  static CUTLASS_HOST_DEVICE void clear(AccessType& dst) {
    uint2 const zero = make_uint2(0, 0);
    dst.registers[0] = zero.x;
    dst.registers[1] = zero.y;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <MemorySpace::Kind Memory_, int kStride>
struct Load<double, 2, Memory_, FragmentElementType::kScalar, double, kStride, 16> {
  /// The output type.
  typedef typename Vectorize<double, 2>::Type AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& dst, double const* pointer, int offset) {
    double2 tmp = reinterpret_cast<double2 const*>(&pointer[offset])[0];
    dst[0] = tmp.x;
    dst[1] = tmp.y;
  }

  /// The clear function.
  static CUTLASS_HOST_DEVICE void clear(AccessType& dst) {
    double2 zero = make_double2(0, 0);
    dst[0] = zero.x;
    dst[1] = zero.y;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDACC_VERSION_MAJOR) && __CUDACC_VERSION_MAJOR < 10
// WAR bug in NVCC where the upper and lower half of the register end up being the same
template <MemorySpace::Kind Memory_, int kStride>
struct Load<half, 8, Memory_, FragmentElementType::kScalar, half, kStride, 16> {
  /// The output type.
  typedef typename Vectorize<half, 8>::Type AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& dst, half const* pointer, int offset) {
    int2 tmp = reinterpret_cast<int2 const*>(&pointer[offset])[0];
    dst.registers[0] = tmp.x;
    dst.registers[1] = tmp.y;

    tmp = reinterpret_cast<int2 const*>(&pointer[offset + 4])[0];
    dst.registers[2] = tmp.x;
    dst.registers[3] = tmp.y;
  }

  /// The clear function.
  static CUTLASS_HOST_DEVICE void clear(AccessType& dst) {
    int2 zero = make_int2(0,0);
    dst.registers[0] = zero.x;
    dst.registers[1] = zero.y;
    dst.registers[2] = zero.x;
    dst.registers[3] = zero.y;
  }
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kAccessSize, MemorySpace::Kind Memory_, int kStride>
struct Load<Scalar_, kAccessSize, Memory_, FragmentElementType::kScalar, Scalar_, kStride, 16> {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& dst, Scalar_ const* pointer, int offset) {
    uint4 tmp = reinterpret_cast<uint4 const*>(&pointer[offset])[0];
    dst.registers[0] = tmp.x;
    dst.registers[1] = tmp.y;
    dst.registers[2] = tmp.z;
    dst.registers[3] = tmp.w;
  }

  /// The clear function.
  static CUTLASS_HOST_DEVICE void clear(AccessType& dst) {
    uint4 zero = make_uint4(0, 0, 0, 0);
    dst.registers[0] = zero.x;
    dst.registers[1] = zero.y;
    dst.registers[2] = zero.z;
    dst.registers[3] = zero.w;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          int kAccessSize,
          MemorySpace::Kind Memory_,
          FragmentElementType::Kind kFragmentElementType = FragmentElementType::kScalar,
          typename FragmentElement_ = Scalar_,
          int kStride = 1,
          size_t size = (sizeof(Scalar_) * kAccessSize)>
struct Store {
  /// The output type.
  typedef typename Vectorize<FragmentElement_, kAccessSize>::Type AccessType;

  /// The store function.
  static CUTLASS_HOST_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    pointer[offset] = *reinterpret_cast<Scalar_ const*>(&src);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kAccessSize, MemorySpace::Kind Memory_>
struct Store<Scalar_, kAccessSize, Memory_, FragmentElementType::kScalar, Scalar_, 1, 2> {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The store function.
  static CUTLASS_HOST_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    uint16_t* addr = reinterpret_cast<uint16_t*>(&pointer[offset]);
    addr[0] = reinterpret_cast<uint16_t const&>(src);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kAccessSize, MemorySpace::Kind Memory_, int kStride>
struct Store<Scalar_, kAccessSize, Memory_, FragmentElementType::kScalar, Scalar_, kStride, 4> {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The store function.
  static CUTLASS_HOST_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    uint32_t* addr = reinterpret_cast<uint32_t*>(&pointer[offset]);
    addr[0] = src.registers[0];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kAccessSize, MemorySpace::Kind Memory_, int kStride>
struct Store<Scalar_, kAccessSize, Memory_, FragmentElementType::kScalar, Scalar_, kStride, 8> {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The store function.
  static CUTLASS_HOST_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    uint2* addr = reinterpret_cast<uint2*>(&pointer[offset]);
    addr[0] = make_uint2(src.registers[0], src.registers[1]);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <MemorySpace::Kind Memory_, int kStride>
struct Store<double, 2, Memory_, FragmentElementType::kScalar, double, kStride, 16> {
  /// The output type.
  typedef typename Vectorize<double, 2>::Type AccessType;

  /// The store function.
  static CUTLASS_HOST_DEVICE void store(AccessType const& src, double* pointer, int offset) {
    double2* addr = reinterpret_cast<double2*>(&pointer[offset]);
    addr[0] = make_double2(src[0], src[1]);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, int kAccessSize, MemorySpace::Kind Memory_, int kStride>
struct Store<Scalar_, kAccessSize, Memory_, FragmentElementType::kScalar, Scalar_, kStride, 16> {
  /// The output type.
  typedef typename Vectorize<Scalar_, kAccessSize>::Type AccessType;

  /// The store function.
  static CUTLASS_HOST_DEVICE void store(AccessType const& src, Scalar_* pointer, int offset) {
    uint4* addr = reinterpret_cast<uint4*>(&pointer[offset]);
    addr[0] = make_uint4(src.registers[0], src.registers[1], src.registers[2], src.registers[3]);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          int kAccessSize,
          MemorySpace::Kind Memory_,
          typename FragmentElement_,
          int kStride,
          size_t size>
struct Load<Scalar_,
            kAccessSize,
            Memory_,
            FragmentElementType::kWmmaMatrix,
            FragmentElement_,
            kStride,
            size> {
  /// The output type.
  typedef FragmentElement_ AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& value, Scalar_ const* pointer, int offset) {
    value.load(&pointer[offset], kStride);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kAccessSize,
          MemorySpace::Kind Memory_,
          typename FragmentElement_,
          int kStride,
          size_t size>
struct Load<Vector<bin1_t, 32>,
            kAccessSize,
            Memory_,
            FragmentElementType::kWmmaMatrix,
            FragmentElement_,
            kStride,
            size> {
  /// The output type.
  typedef FragmentElement_ AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& value, Vector<bin1_t, 32> const* pointer,
                                       int offset) {
    value.load(&pointer[offset], kStride * 32);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kAccessSize,
          MemorySpace::Kind Memory_,
          typename FragmentElement_,
          int kStride,
          size_t size>
struct Load<Vector<int4_t, 8>,
            kAccessSize,
            Memory_,
            FragmentElementType::kWmmaMatrix,
            FragmentElement_,
            kStride,
            size> {
  /// The output type.
  typedef FragmentElement_ AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& value, Vector<int4_t, 8> const* pointer,
                                       int offset) {
    value.load(&pointer[offset], kStride * 8);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kAccessSize,
          MemorySpace::Kind Memory_,
          typename FragmentElement_,
          int kStride,
          size_t size>
struct Load<Vector<uint4_t, 8>,
            kAccessSize,
            Memory_,
            FragmentElementType::kWmmaMatrix,
            FragmentElement_,
            kStride,
            size> {
  /// The output type.
  typedef FragmentElement_ AccessType;

  /// The load function.
  static CUTLASS_HOST_DEVICE void load(AccessType& value, Vector<uint4_t, 8> const* pointer,
                                       int offset) {
    value.load(&pointer[offset], kStride * 8);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar_,
          int kAccessSize,
          MemorySpace::Kind Memory_,
          typename FragmentElement_,
          int kStride,
          size_t size>
struct Store<Scalar_,
             kAccessSize,
             Memory_,
             FragmentElementType::kWmmaMatrix,
             FragmentElement_,
             kStride,
             size> {
  /// The input type.
  typedef FragmentElement_ AccessType;

  /// The store function.
  static CUTLASS_HOST_DEVICE void store(AccessType const& value, Scalar_* pointer, int offset) {
    value.store(&pointer[offset], kStride);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
