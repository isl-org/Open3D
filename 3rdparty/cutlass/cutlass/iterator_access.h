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
    \brief Free functions for loading and storing to implementations of tile iteartor concepts.
*/
#pragma once

#include "cutlass/load_store.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/shape.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Used by convolution
template <typename InputIterator, typename Fragment>
CUTLASS_HOST_DEVICE void iterator_load(InputIterator &iterator, Fragment &fragment) {
  typename InputIterator::FragmentIterator frag_iterator(fragment);
  CUTLASS_PRAGMA_UNROLL
  for (int d = 0; d < InputIterator::Iterations::kD; ++d) {
    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < InputIterator::Iterations::kH; ++h) {
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < InputIterator::Iterations::kW; ++w) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < InputIterator::Iterations::kC; ++c) {
          if (iterator.valid(d, h, w, c)) {
            iterator.load_element(reinterpret_cast<typename InputIterator::AccessType &>(
                                      frag_iterator.at(d, h, w, c)),
                                  d,
                                  h,
                                  w,
                                  c);
          }
        }
        if (w < InputIterator::Iterations::kW - 1) {
          iterator.inc_w();
        }
      }
      if (h < InputIterator::Iterations::kH - 1) {
        iterator.inc_h();
      }
    }
    if (d < InputIterator::Iterations::kD - 1) {
      iterator.inc_d();
    }
  }
  iterator.inc_advance();
}

template <typename OutputIterator, typename Fragment>
CUTLASS_HOST_DEVICE void iterator_store(OutputIterator &iterator, Fragment &fragment) {
  typename OutputIterator::FragmentIterator frag_iterator(fragment);
  CUTLASS_PRAGMA_UNROLL
  for (int d = 0; d < OutputIterator::Iterations::kD; ++d) {
    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < OutputIterator::Iterations::kH; ++h) {
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < OutputIterator::Iterations::kW; ++w) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < OutputIterator::Iterations::kC; ++c) {
          if (iterator.valid(d, h, w, c)) {
            iterator.store_element(reinterpret_cast<typename OutputIterator::AccessType &>(
                                       frag_iterator.at(d, h, w, c)),
                                   d,
                                   h,
                                   w,
                                   c);
          }
        }
        if (w < OutputIterator::Iterations::kW - 1) {
          iterator.inc_w();
        }
      }
      if (h < OutputIterator::Iterations::kH - 1) {
        iterator.inc_h();
      }
    }
    if (d < OutputIterator::Iterations::kD - 1) {
      iterator.inc_d();
    }
  }
  iterator.inc_advance();
}
////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
