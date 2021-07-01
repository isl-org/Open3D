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

#include <functional>
#include <iosfwd>
#include <iostream>
#include <vector>

#include "cutlass/tensor_view.h"

#include "tools/util/half.h"
#include "tools/util/host_tensor_view.h"
#include "tools/util/tensor_view_io.h"
#include "tools/util/type_traits.h"

namespace test {

/// Defines an arrangement
class Layout {
 public:
  /// Analogous to cutlass::Span
  struct Span {
    int dim;
    int size;

    Span(int _dim = 0, int _size = 0) : dim(_dim), size(_size) {}
  };

  /// Vector of span definitions
  typedef std::vector<Span> SpanVector;

  /// Coordinate in an arbitrary-dimensional space
  typedef std::vector<int> Coordinate;

  /// Defines a vector describing the extent of some dimension
  typedef std::vector<int> ExtentVector;

 private:
  /// Defines a mapping from a 1D sequence to an n-dimensional space
  SpanVector layout_;

  /// Computes the extent of each node in the layout description
  ExtentVector extent_;

  /// For each dimension, computes extent
  std::vector<ExtentVector> dim_extent_;

 public:
  //
  // Methods
  //

  Layout();
  Layout(SpanVector const& _layout);

  /// Updates the layout
  void reset(SpanVector const& _layout = SpanVector());

  /// Computes the rank of the layout
  int rank() const;

  /// Prints Layout data structure
  std::ostream& write(std::ostream& out) const;

  /// Maps an index to a given coordinate
  Coordinate operator()(int index) const;

  /// Maps a coordinate to an index
  int operator()(Coordinate const& coord) const;
};
}

/// Implemented in layout_verification.cu
namespace std {
std::ostream& operator<<(std::ostream& out, test::Layout::Coordinate const& coord);
}

namespace test {

/// Packs the elements of a coordinate into the bits available
template <typename T, int Rank = 2>
struct CoordinatePack {
  typedef T value_type;

  typedef typename cutlass::TypeTraits<T>::unsigned_type Bits;

  static int const ElementBits = sizeof(Bits) * 8 / Rank;

  static Bits const Mask = (Bits(1) << ElementBits) - 1;

  Bits operator()(Layout::Coordinate const& coord) const {
    Bits result = 0;
    for (size_t i = 0; i < coord.size(); ++i) {
      result |= (((coord.at(i) & Mask) << (i * ElementBits)));
    }

    return result;
  }
};

/// Unpacks a coordinate from the bits available
template <typename T, int Rank = 2>
struct CoordinateUnpack {
  typedef T value_type;

  typedef typename cutlass::TypeTraits<T>::unsigned_type Bits;

  static int const ElementBits = sizeof(Bits) * 8 / Rank;

  static Bits const Mask = (Bits(1) << ElementBits) - 1;

  Layout::Coordinate operator()(Bits index) const {
    Layout::Coordinate coord(Rank, 0);

    for (size_t i = 0; i < Rank; ++i) {
      coord.at(i) = (index & Mask);
      index = (index >> ElementBits);
    }

    return coord;
  }
};

/// Hashing function
struct HashUint64 {
  // PJW Elf Hash - https://en.wikipedia.org/wiki/PJW_hash_function
  uint64_t operator()(uint64_t value) const {
    uint64_t h = 0;
    uint64_t high;
    uint8_t const* s = reinterpret_cast<uint8_t const*>(&value);

    for (int byte = 0; byte < sizeof(value); ++byte) {
      h = (h << 4) + *s++;
      if (high = (h & 0xF0000000)) {
        h ^= high >> 24;
      }
      h &= ~high;
    }
    return h;
  }
};

/// Packs the coordinate into 64 bits then hashes the result and stores the least significant bits
template <typename T, typename Hasher = HashUint64>
struct CoordinateHash {
  typedef T value_type;

  typedef typename cutlass::TypeTraits<T>::unsigned_type Bits;

  typedef CoordinatePack<uint64_t, 4> Pack;

  /// Bit mask to cast from uint64_t to whatever Bits is
  static uint64_t const mask = ((uint64_t(1) << (sizeof(Bits) * 8)) - 1);

  static_assert(sizeof(Bits) <= sizeof(uint64_t), "T must be smaller than or equal to uint64_t");

  //
  // Data members
  //

  /// Packs coordinate into uint64_t
  Pack pack;

  /// Hashes the resulting coordinate
  Hasher hasher;

  /// One additional xor to salt things
  uint64_t salt;

  //
  //
  //

  CoordinateHash(uint64_t _salt = 0x0ac7d0190) : salt(_salt) {}

  /// Returns a hashed coordinate
  Bits operator()(Layout::Coordinate const& coord) const {
    uint64_t result = hasher(pack(coord) ^ salt);

    return Bits(result & mask);
  }
};

/// Environment to initialize and verify a template
template <typename DestType_,
          typename DestCoordinateHash_,
          typename SourceType_,
          typename SourceCoordinateHash_>
class VerifyLayout {
 public:
  typedef DestType_ DestType;

  typedef typename cutlass::TypeTraits<DestType>::unsigned_type DestBits;

  typedef DestCoordinateHash_ DestCoordinateHash;

  typedef SourceType_ SourceType;

  typedef typename cutlass::TypeTraits<SourceType>::unsigned_type SourceBits;

  typedef SourceCoordinateHash_ SourceCoordinateHash;

 public:
  /// Basic visitor to terminate verification on error
  struct VisitorNop {
    /// Returns true to keep checking in spite of errors, false if to stop
    bool operator()(DestBits got,              // hashed/packed coordinate encountered
                    DestBits expected,         // hashed/packed coordinate expected
                    Layout::Coordinate coord,  // computed coordinate
                    int index) {               // location

      // false to terminate checking
      return false;
    }
  };

  /// Basic visitor to terminate verification on error
  struct VisitorVerbose {
    CoordinateUnpack<DestBits> unpack;

    std::ostream* out;

    VisitorVerbose() : out(&std::cout) {}
    VisitorVerbose(std::ostream& _out) : out(&_out) {}

    /// Returns true to keep checking in spite of errors, false if to stop
    bool operator()(DestBits got,              // hashed/packed coordinate encountered
                    DestBits expected,         // hashed/packed coordinate expected
                    Layout::Coordinate coord,  // computed coordinate
                    int index) {               // location

      int const width = sizeof(DestBits) * 2;

      (*out) << "[" << index << "] - (" << coord << ") - expected: 0x" << std::hex
             << std::setw(width) << std::setfill('0') << expected << ", got: 0x" << std::setw(width)
             << std::setfill('0') << got << std::dec << " - unpacked: (" << unpack(got) << ")"
             << std::endl;

      // true to print out complete error report
      return true;
    }
  };

 public:
  VerifyLayout() {}

  /// Initializes memory according to a layout and hash function
  void initialize(cutlass::HostTensorView<SourceType> const& source, Layout const& layout) {
    SourceCoordinateHash hash;

    int const count = source.size().count();

    SourceBits* data = reinterpret_cast<SourceBits*>(source.ref().data());
    for (int index = 0; index < count; ++index) {
      SourceBits element = hash(layout(index));

      // std::cout << "  " << index << ": 0x" << std::hex << element << std::dec << std::endl;

      data[index] = element;
    }
  }

  /// Verifies the resulting layout
  template <typename Visitor>
  bool verify(cutlass::HostTensorView<DestType> const& dest,
              Layout const& layout,
              Visitor visitor) {
    DestCoordinateHash hash;

    int const count = dest.size().count();

    DestBits* data = reinterpret_cast<DestBits*>(dest.ref().data());

    int errors = 0;
    for (int index = 0; index < count; ++index) {
      Layout::Coordinate coord = layout(index);
      DestBits element = hash(coord);
      if (element != data[index]) {
        ++errors;
        if (!visitor(data[index], element, coord, index)) {
          break;
        }
      }
    }

    return !errors;
  }

  /// Verifies the resulting layout
  bool verify(cutlass::HostTensorView<DestType> const& dest, Layout const& layout) {
    return verify(layout, VisitorNop());
  }
};

}  // namespace test
