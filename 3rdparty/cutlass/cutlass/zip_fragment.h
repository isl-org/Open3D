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
    \brief Models a pair of fragments
*/
#pragma once

#include <assert.h>

#include "cutlass/cutlass.h"
#include "cutlass/shape.h"
#include "cutlass/util/cutlass_math.h"
#include "cutlass/vector.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief A template defining \ref fragment_concept
* @concept{fragment_concept}
*/
template <typename First_, typename Second_>
struct ZipFragment {
  /// First fragment object
  typedef First_ First;

  /// Second fragment object
  typedef Second_ Second;

  /// This class.
  typedef ZipFragment<First, Second> This_;

  //
  // Data members
  //

  /// First fragment object
  First first;

  /// Second fragment object
  Second second;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_DEVICE
  ZipFragment() { }

  /// Copy ctor
  CUTLASS_DEVICE
  ZipFragment(First const &_first, Second const &_second): first(_first), second(_second) { }

  /// Clear a fragment.
  CUTLASS_DEVICE void clear() {
    first.clear();
    second.clear();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to construct a ZipFragment object
template <typename First, typename Second>
CUTLASS_HOST_DEVICE
ZipFragment<First, Second> make_ZipFragment(First const &first, Second const &second) {
  return ZipFragment<First, Second>(first, second);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Zips two convert operations
template <typename First_, typename Second_>
struct ZipConvert {
  /// First convert operator
  typedef First_ First;

  /// Second convert operator
  typedef Second_ Second;

  /// Defines the input zip fragment
  typedef ZipFragment<typename First::InputFragment, typename Second::InputFragment> InputFragment;

  /// Defines the output zip fragment
  typedef ZipFragment<typename First::OutputFragment, typename Second::OutputFragment>
      OutputFragment;

  //
  //
  //

  /// First transformer
  First first;

  /// Second transformer
  Second second;

  //
  //
  //

  /// Ctor.
  CUTLASS_DEVICE ZipConvert() {}

  /// Ctor.
  CUTLASS_DEVICE ZipConvert(First const &_first, Second const &_second): first(_first), second(_second) { }

  /// Transform a fragment.
  CUTLASS_DEVICE void transform(InputFragment const& src, OutputFragment& dst) {
    first.transform(src.first, dst.first);
    second.transform(src.second, dst.second);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to construct a ZipConvert object
template <typename First, typename Second>
CUTLASS_HOST_DEVICE
ZipConvert<First, Second> make_ZipConvert(First const &first, Second const &second) {
  return ZipConvert<First, Second>(first, second);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
