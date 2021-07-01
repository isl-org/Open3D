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
#pragma once

/*! \file
    \brief This header contains a class to parametrize a statistical distribution function.
*/

#include <fstream>

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Distribution type
struct Distribution {
  /// Variant types
  enum Kind { Invalid, Uniform, Gaussian, Linear, Identity };

  /// Distribution state
  union {
    /// Uniform distribution
    struct {
      double min;
      double max;
    } uniform;

    /// Gaussian distribution
    struct {
      double mean;
      double stddev;
    } gaussian;

    /// Elements are linear combination of row and column index
    struct {
      double offset;
      double delta_row;
      double delta_column;
    } linear;
  };

  /// Active variant kind
  Kind kind;

  /// Random values are cast to integer after scaling by this power of two
  int int_scale;

  //
  // Methods
  //

  Distribution() : kind(Invalid), int_scale(0) {}

  /// Configures distribution as uniform random
  Distribution &set_uniform(double _min, double _max, int _int_scale = 0) {
    kind = Uniform;
    uniform.min = _min;
    uniform.max = _max;
    int_scale = _int_scale;
    return *this;
  }

  /// Configures distribution as Gaussian distribution
  Distribution &set_gaussian(double _mean, double _stddev, int _int_scale = 0) {
    kind = Gaussian;
    gaussian.mean = _mean;
    gaussian.stddev = _stddev;
    int_scale = _int_scale;
    return *this;
  }

  /// Sets identity
  Distribution &set_identity() {
    kind = Identity;
    return *this;
  }

  /// Configures distribution as linear combination of row and column index
  Distribution &set_linear(double _offset, double _delta_row, double _delta_column) {
    kind = Linear;
    linear.offset = _offset;
    linear.delta_row = _delta_row;
    linear.delta_column = _delta_column;
    return *this;
  }
};

}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Prints a Distribution to ostream
inline std::ostream &operator<<(std::ostream &out, cutlass::Distribution const &dist) {
  switch (dist.kind) {
    case cutlass::Distribution::Uniform:
      out << "uniform, min: " << dist.uniform.min << ", max: " << dist.uniform.max;
      break;
    case cutlass::Distribution::Gaussian:
      out << "gaussian, mean: " << dist.gaussian.mean << ", stddev: " << dist.gaussian.stddev;
      break;
    case cutlass::Distribution::Linear:
      out << "linear, mean: " << dist.linear.offset << ", delta_row: " << dist.linear.delta_row
          << ", delta_column: " << dist.linear.delta_column;
      break;
    case cutlass::Distribution::Identity:
      break;
    default:
      out << "unknown";
  }

  out << ", int_scale: " << dist.int_scale;

  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
