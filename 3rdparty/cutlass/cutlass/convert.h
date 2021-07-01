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
/*!
  \file
  \brief Defines conversion operations among Fragments of different base type.
*/
#pragma once

#include "cutlass/fragment.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputFragment_, typename OutputFragment_>
struct Convert {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputScalar_, typename OutputScalar_, int kScalars_>
struct Convert<Fragment<InputScalar_, kScalars_>, Fragment<OutputScalar_, kScalars_> > {
  /// The input fragment.
  typedef Fragment<InputScalar_, kScalars_> InputFragment;
  /// The output fragment.
  typedef Fragment<OutputScalar_, kScalars_> OutputFragment;

  /// Ctor.
  CUTLASS_DEVICE Convert() {}

  /// Transform a fragment.
  CUTLASS_DEVICE void transform(InputFragment const& src, OutputFragment& dst) {
    transform(src, 0, dst);
  }

  /// Transform a fragment.
  template <typename Fragment_>
  CUTLASS_DEVICE void transform(Fragment_ const& src, int offset, OutputFragment& dst) {
    for (int i = 0; i < kScalars_; ++i) {
      dst[i] = static_cast<OutputScalar_>(src[i + offset]);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment_>
struct Copy {
  /// The input fragment.
  typedef Fragment_ InputFragment;
  /// The output fragment.
  typedef Fragment_ OutputFragment;

  /// Ctor.
  CUTLASS_DEVICE Copy() {}

  /// Transform a fragment.
  CUTLASS_DEVICE void transform(Fragment_ const& src, Fragment_& dst) { transform(src, 0, dst); }

  /// Transform a fragment.
  template <typename InputFragment_>
  CUTLASS_DEVICE void transform(InputFragment_ const& src, int offset, Fragment_& dst) {
    if (sizeof(typename Fragment_::Element) == 8) {
      uint64_t const* src_ptr = reinterpret_cast<uint64_t const*>(&src[offset]);
      uint64_t* dst_ptr = reinterpret_cast<uint64_t*>(&dst[0]);
      for (int i = 0; i < sizeof(Fragment_) / 8; ++i) {
        dst_ptr[i] = src_ptr[i];
      }
    } else {
      uint32_t const* src_ptr = reinterpret_cast<uint32_t const*>(&src[offset]);
      uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(&dst[0]);
      for (int i = 0; i < sizeof(Fragment_) / 4; ++i) {
        dst_ptr[i] = src_ptr[i];
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
