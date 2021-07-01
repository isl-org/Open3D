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
    \brief Defines properties of matrices used to denote layout and operands to GEMM kernels.
*/
#pragma once

#include "cutlass/coord.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// MatrixCoord wraps Coord<2, int> to provide a helper for accessing named dimensions. Classes
/// expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.
struct MatrixCoord : public Coord<2, int> {

  /// Integer-valued index
  typedef int Index;

  /// Base type is a Coord of rank=2
  typedef Coord<2, Index> Base;

  /// Rows dimension
  static int const kRow = 0;

  /// Columns dimension
  static int const kColumn = 1;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  MatrixCoord() { }

  /// Constructs from Coord<2>
  CUTLASS_HOST_DEVICE
  MatrixCoord(Coord<2, Index> const &coord): Base(coord) { }

  /// Helper to construct from a row and column
  CUTLASS_HOST_DEVICE
  MatrixCoord(Index row, Index column): Base(make_Coord(row, column)) { }

  /// Returns the row of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & row() const { return this->at(kRow); }

  /// Returns the row of the coordinate
  CUTLASS_HOST_DEVICE
  Index & row() { return this->at(kRow); }

  /// Returns the column of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & column() const { return this->at(kColumn); }

  /// Returns the column of the coordinate
  CUTLASS_HOST_DEVICE
  Index & column() { return this->at(kColumn); }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  MatrixCoord operator+(Base const& b) const {
    return MatrixCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  MatrixCoord operator-(Base const& b) const {
    return MatrixCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  MatrixCoord operator*(Base const& b) const {
    return MatrixCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  MatrixCoord operator/(Base const& b) const {
    return MatrixCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  MatrixCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  MatrixCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  MatrixCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  MatrixCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines data layouts of various matrix formats usable by TensorRef and other classes.
//
// The following define classes satisfying the TensorRefMapFunc concept. These must support the
// following operations, where func is an instance of type TensorRefMapFunc.
//
//   Coord<TensorRefMapFunc::kStorageRank> = func(Coord<kRank>);
//
// Though not required to be usable by TensorRef, each of the following also define a helper
// function to map the "leading dimension" to an appropriate stride vector. Implementations
// following this convention should also implement the following static method:
//
//   Coord<TensorRefMapFunc::kStorageRank> stride = TensorRefMapFunc::stride(leading_dim);
//
namespace MatrixLayout {

  /// Enumeration defining fundamental contiguous layouts.
  enum Kind { kRowMajor, kColumnMajor };

  //
  // TensorRefMapFunc definitions for common layouts
  //

  /// Mapping function for row-major matrices
  struct RowMajor {
    static int const kStorageRank = 2;
    /// Maps (i, j) to (i, j)
    CUTLASS_HOST_DEVICE
    Coord<kStorageRank> operator()(MatrixCoord const &coord) const {
      return coord;
    }
  };

  /// Mapping function for column-major matrices
  struct ColumnMajor {
    static int const kStorageRank = 2;
    /// Maps (i, j) to (j, i)
    CUTLASS_HOST_DEVICE
    Coord<kStorageRank> operator()(MatrixCoord const &coord) const {
      return make_Coord(coord.column(), coord.row());
    }
  };

  /// Mapping function for interleaved matrices. Matrix is structured
  /// as row-major arrangement of fixed-size columns.
  template <int Interleave>
  struct RowMajorInterleaved {

    /// Rank of storage n-D array
    static int const kStorageRank = 3;

    /// Interleaving size
    static int const kInterleave = Interleave;

    /// Maps (row, col) to (row, col, row)
    CUTLASS_HOST_DEVICE
    Coord<kStorageRank> operator()(MatrixCoord const &coord) const {
      return make_Coord(
        coord.row() / kInterleave,
        coord.column(),
        coord.row() % kInterleave
      );
    }

    /// Helper to compute stride vector from leading dimension
    CUTLASS_HOST_DEVICE
    static Coord<kStorageRank> stride(int ldm) {
      return make_Coord(
        ldm * kInterleave,
        kInterleave,
        1
      );
    }
  };

  /// Mapping function for interleaved matrices. Matrix is structured
  /// as column-major arrangement of fixed-size rows.
  template <int Interleave>
  struct ColumnMajorInterleaved {

    /// Rank of storage n-D array
    static int const kStorageRank = 3;

    /// Interleaving size
    static int const kInterleave = Interleave;

    /// Maps (row, col) to (col, row, col)
    CUTLASS_HOST_DEVICE
    Coord<kStorageRank> operator()(MatrixCoord const &coord) const {
      return make_Coord(
        coord.column() / kInterleave,
        coord.row(),
        coord.column() % kInterleave
      );
    }

    /// Helper to compute stride vector from leading dimension
    CUTLASS_HOST_DEVICE
    static Coord<kStorageRank> stride(int ldm) {
      return make_Coord(
        ldm * kInterleave,
        kInterleave,
        1
      );
    }
  };

  /// Mapping function for scenario in which layout is row-major or column-major but this information
  /// is only available at runtime.
  struct ContiguousLayout {
    /// Arbitrary storage rank
    static int const kStorageRank = 3;

    /// Dimension of rows
    static int const kRow = 0;

    /// Dimension of columns
    static int const kColumn = 1;

    /// Mapping function defined by runtime variable. Returns coordinates in n-D storage array
    /// as (matrix row, matrix colum, 0)
    CUTLASS_HOST_DEVICE
    Coord<kStorageRank> operator()(MatrixCoord const &coord) const {
        return make_Coord(coord.row(), coord.column(), 0);
    }

    /// Helper to construct a stride vector based on contiguous matrix layout and leading dimension
    CUTLASS_HOST_DEVICE
    static Coord<kStorageRank> stride(MatrixLayout::Kind layout, int ldm) {
      if (layout == MatrixLayout::kRowMajor) {
        return make_Coord(ldm, 1, 1);
      }
      return make_Coord(1, ldm, 1);
    }
  };

  /// Mapping function for block-linear matrices. Matrix is structured
  /// as column-major arrangement of 2D tiles (that are column-major).
  template <int BlockRows, int BlockColumns>
  struct ColumnMajorBlockLinear {

    /// Rank of storage n-D array
    static int const kStorageRank = 4;

    /// Interleaving size in rows dimension
    static int const kBlockRows = BlockRows;

    /// Interleaving size in columns dimension
    static int const kBlockColumns = BlockColumns;

    /// Maps (row, col) to (col, row, col, row)
    CUTLASS_HOST_DEVICE
    Coord<kStorageRank> operator()(MatrixCoord const &coord) const {
      return make_Coord(
        coord.column() / kBlockColumns,
        coord.row() / kBlockRows,
        coord.column() % kBlockColumns,
        coord.row() % kBlockRows
      );
    }

    /// Helper to compute stride vector from leading dimension
    CUTLASS_HOST_DEVICE
    static Coord<kStorageRank> stride(int ldm) {
      return make_Coord(
        ldm * kBlockRows * kBlockColumns,
        kBlockRows * kBlockColumns,
        kBlockRows,
        1
      );
    }
  };

  /// Mapping function for block-linear matrices. Matrix is structured
  /// as row-major arrangement of 2D tiles (that are row-major)
  template <int BlockRows, int BlockColumns>
  struct RowMajorBlockLinear {

    /// Rank of storage n-D array
    static int const kStorageRank = 4;

    /// Interleaving size in rows dimension
    static int const kBlockRows = BlockRows;

    /// Interleaving size in columns dimension
    static int const kBlockColumns = BlockColumns;

    /// Maps (row, col) to (row, col, row, col)
    CUTLASS_HOST_DEVICE
    Coord<kStorageRank> operator()(MatrixCoord const &coord) const {
      return make_Coord(
        coord.row() / kBlockRows,
        coord.column() / kBlockColumns,
        coord.row() % kBlockRows,
        coord.column() % kBlockColumns
      );
    }

    /// Helper to compute stride vector from leading dimension
    CUTLASS_HOST_DEVICE
    static Coord<kStorageRank> stride(int ldm) {
      return make_Coord(
        ldm * kBlockRows * kBlockColumns,
        kBlockRows * kBlockColumns,
        kBlockColumns,
        1
      );
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemm operand - D = A * B + C
struct GemmOperand {
  enum Kind { kA, kB, kC, kD };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Transformation applied to matrix operands
struct MatrixTransform {
  enum Kind {
    kNone,       /// no operation
    kConjugate,  /// conjugate
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Tensor layout
namespace TensorLayout {

  enum Kind { kNHWC, kNCHW };
};
////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
