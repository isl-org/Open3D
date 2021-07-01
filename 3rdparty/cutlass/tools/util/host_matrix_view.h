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
  \brief HostMatrix is a helper to define a HostTensor of rank=2 with a contiguous layout.

  See tools/util/host_tensor.h for more details.
*/

#include "cutlass/matrix_traits.h"
#include "tools/util/host_tensor.h"

#include "tools/util/reference/host/gemm.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to define a rank=2 host matrix with contiguous layout
template <
  typename T
>
class HostMatrixView :
  public HostTensorView<T, 2, MatrixLayout::ContiguousLayout, 3, int> {
public:

  /// Base class is a HostTensor of rank=2 with contiguous layout
  typedef HostTensorView<T, 2, MatrixLayout::ContiguousLayout, 3, int> Base;

  /// Tensor coordinate
  typedef typename Base::TensorCoord TensorCoord;

  /// Index type
  typedef typename Base::Index Index;

private:

  /// Layout of contiguous matrix
  MatrixLayout::Kind layout_;

public:

  /// Default ctor
  HostMatrixView(): layout_(MatrixLayout::kColumnMajor) { }

  /// Constructs a HostTensor from size. Assumes column-major and infers leading dimension
  HostMatrixView(TensorCoord const& size): layout_(MatrixLayout::kColumnMajor) {
    Index ldm = size[0];
    this->reset(MatrixLayout::ContiguousLayout::stride(layout_, ldm), size);
  }

  /// Constructs a HostTensor from size and layout - infers leading dimension
  HostMatrixView(TensorCoord const& size, MatrixLayout::Kind layout): layout_(layout) {
    Index ldm = (layout_ == MatrixLayout::kColumnMajor ? size[0] : size[1]);
    this->reset(MatrixLayout::ContiguousLayout::stride(layout_, ldm), size);
  }

  /// Constructs a HostTensor given size, layout, and leading dimension
  HostMatrixView(TensorCoord const& size, Index ldm, MatrixLayout::Kind layout): layout_(layout) {
    this->reset(MatrixLayout::ContiguousLayout::stride(layout_, ldm), size);
  }

  /// Gets the leading dimension of the matrix
  Index leading_dim() const {
    if (layout_ == MatrixLayout::kColumnMajor) {
      return this->stride(MatrixLayout::ContiguousLayout::kColumn);
    }
    else {
      return this->stride(MatrixLayout::ContiguousLayout::kRow);
    }
  }

  /// Returns contiguous matrix layout kind
  MatrixLayout::Kind get_layout() const {
    return layout_;
  }

  /// Returns size as a MatrixCoord
  MatrixCoord size() const {
    return MatrixCoord(Base::size());
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to define a rank=2 host matrix with column-major layout
template <typename T>
class HostMatrixViewColumnMajor :
  public HostTensorView<T, 2, MatrixLayout::ColumnMajor, 2, int, long long> {
public:

  /// Base class is a HostTensorView of rank=2 with contiguous layout
  typedef HostTensorView<T, 2, MatrixLayout::ColumnMajor, 2, int, long long> Base;

  /// Tensor coordinate
  typedef typename Base::TensorCoord TensorCoord;

  /// Index type
  typedef typename Base::Index Index;

public:

  /// Default ctor
  HostMatrixViewColumnMajor() { }

  /// Constructs a HostMatrixViewColumnMajor from size. Assumes column-major and infers leading dimension
  HostMatrixViewColumnMajor(TensorCoord const& size): Base(size, size[0]) {

  }

  /// Constructs a HostMatrixViewColumnMajor given size, layout, and leading dimension
  HostMatrixViewColumnMajor(TensorCoord const& size, Index ldm) {
    this->reset(make_Coord(ldm, 1), size);
  }

  /// Returns contiguous matrix layout kind
  MatrixLayout::Kind get_layout() const {
    return MatrixLayout::kColumnMajor;
  }

  /// Gets the leading dimension of the matrix
  Index leading_dim() const {
    return this->stride(0);
  }

  /// Returns size as a MatrixCoord
  MatrixCoord size() const {
    return MatrixCoord(Base::size());
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to define a rank=2 host matrix with row-major layout
template <typename T>
class HostMatrixViewRowMajor :
  public HostTensorView<T, 2, MatrixLayout::RowMajor, 2, int, long long> {
public:

  /// Base class is a HostTensor of rank=2 with contiguous layout
  typedef HostTensorView<T, 2, MatrixLayout::RowMajor, 2, int, long long> Base;

  /// Tensor coordinate
  typedef typename Base::TensorCoord TensorCoord;

  /// Index type
  typedef typename Base::Index Index;

public:

  /// Default ctor
  HostMatrixViewRowMajor() { }

  /// Constructs a HostMatrixViewRowMajor from size. Assumes column-major and infers leading dimension
  HostMatrixViewRowMajor(TensorCoord const& size): Base(size, size[1]) {

  }

  /// Constructs a HostMatrixViewRowMajor given size, layout, and leading dimension
  HostMatrixViewRowMajor(TensorCoord const& size, Index ldm) {
    this->reset(make_Coord(ldm, 1), size);
  }

  /// Returns contiguous matrix layout kind
  MatrixLayout::Kind get_layout() const {
    return MatrixLayout::kRowMajor;
  }

  /// Gets the leading dimension of the matrix
  Index leading_dim() const {
    return this->stride(0);
  }

  /// Returns size as a MatrixCoord
  MatrixCoord size() const {
    return MatrixCoord(Base::size());
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
