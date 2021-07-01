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
class HostMatrix :
  public HostTensor<T, 2, MatrixLayout::ContiguousLayout, 3, int, long long> {
public:

  /// Base class is a HostTensor of rank=2 with contiguous layout
  typedef HostTensor<T, 2, MatrixLayout::ContiguousLayout, 3, int, long long> Base;

  /// Index type
  typedef typename Base::Index Index;

private:

  /// Layout of contiguous matrix
  MatrixLayout::Kind layout_;

public:

  /// Default ctor
  HostMatrix(): layout_(MatrixLayout::kColumnMajor) { }

  /// Constructs a HostTensor from size. Assumes column-major and infers leading dimension
  HostMatrix(MatrixCoord const& size, bool _device_backed = true): layout_(MatrixLayout::kColumnMajor) {
    Index ldm = size[0];
    this->reset(MatrixLayout::ContiguousLayout::stride(layout_, ldm), size, _device_backed);
  }

  /// Constructs a HostTensor from size and layout - infers leading dimension
  HostMatrix(MatrixCoord const& size, MatrixLayout::Kind layout, bool _device_backed = true): layout_(layout) {
    Index ldm = (layout_ == MatrixLayout::kColumnMajor ? size[0] : size[1]);
    this->reset(MatrixLayout::ContiguousLayout::stride(layout_, ldm), size, _device_backed);
  }

  /// Constructs a HostTensor given size, layout, and leading dimension
  HostMatrix(MatrixCoord const& size, Index ldm, MatrixLayout::Kind layout, bool _device_backed = true): layout_(layout) {
    this->reset(MatrixLayout::ContiguousLayout::stride(layout_, ldm), size, _device_backed);
  }

  /// Returns contiguous matrix layout kind
  MatrixLayout::Kind get_layout() const {
    return layout_;
  }

  /// Resizes a matrix
  void resize(MatrixCoord const &_size, MatrixLayout::Kind layout, Index ldm = 0, bool _device_backed = true) {
    if (!ldm) {
      ldm = (layout == MatrixLayout::kColumnMajor ? _size[0] : _size[1]);
    }
    layout_ = layout;
    this->reset(MatrixLayout::ContiguousLayout::stride(layout_, ldm), _size, _device_backed);
  }

  /// Helper to resize matrix
  void resize(Index rows, Index columns, MatrixLayout::Kind layout, Index ldm = 0, bool _device_backed = true) {
    this->resize(MatrixCoord(rows, columns), layout, ldm,_device_backed);
  }

  /// Helper to resize matrix
  void resize_matrix(Index rows, Index columns, MatrixLayout::Kind layout, Index ldm = 0, bool _device_backed = true) {
    this->resize(MatrixCoord(rows, columns), layout, ldm,_device_backed);
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

  /// Returns size as a MatrixCoord
  MatrixCoord size() const {
    return MatrixCoord(Base::size());
  }

  /// Returns size in the given dimension
  Index size(int idx) const {
    return Base::size(idx);
  }

  /// Helper to call GEMM operation on HostMatrix objects that differ only in their scalar type.
  template <typename A, typename B, typename Ctype, typename Stype>
  void gemm(
    HostMatrix<A> const& tensor_a,
    HostMatrix<B> const& tensor_b,
    Stype alpha = Stype(1),
    Stype beta = Stype(0)) {

    gemm::GemmCoord problem_size(
      tensor_a.size().column(),
      this->size().column(),
      this->size().row(),
      1);

    cutlass::reference::host::Gemm(
      problem_size,
      alpha,
      tensor_a,
      tensor_b,
      beta,
      *this,
      Ctype(0));
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to define a rank=2 host matrix with column-major layout
template <
  typename T
>
class HostMatrixColumnMajor :
  public HostTensor<T, 2, MatrixLayout::ColumnMajor, 2, int, long long> {
public:

  /// Base class is a HostTensor of rank=2 with contiguous layout
  typedef HostTensor<T, 2, MatrixLayout::ColumnMajor, 2, int, long long> Base;

  /// Tensor coordinate
  typedef typename Base::TensorCoord TensorCoord;

  /// Index type
  typedef typename Base::Index Index;

public:

  /// Default ctor
  HostMatrixColumnMajor() { }

  /// Constructs a HostMatrixColumnMajor from size. Assumes column-major and infers leading dimension
  HostMatrixColumnMajor(TensorCoord const& size, bool _device_backed = true): Base(size, size[0], _device_backed) {

  }

  /// Constructs a HostMatrixColumnMajor given size, layout, and leading dimension
  HostMatrixColumnMajor(TensorCoord const& size, Index ldm, bool _device_backed = true) {
    this->reset(make_Coord(ldm, 1), size, _device_backed);
  }

  /// Resizes a matrix
  void resize(MatrixCoord const &size, int ldm = 0, bool _device_backed = true) {
    this->reset(ldm, size, _device_backed);
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
template <
  typename T
>
class HostMatrixRowMajor :
  public HostTensor<T, 2, MatrixLayout::RowMajor, 2, int, long long> {
public:

  /// Base class is a HostTensor of rank=2 with contiguous layout
  typedef HostTensor<T, 2, MatrixLayout::RowMajor, 2, int, long long> Base;

  /// Tensor coordinate
  typedef typename Base::TensorCoord TensorCoord;

  /// Index type
  typedef typename Base::Index Index;

public:

  /// Default ctor
  HostMatrixRowMajor() { }

  /// Constructs a HostTensor from size. Assumes column-major and infers leading dimension
  HostMatrixRowMajor(TensorCoord const& size, bool _device_backed = true) {
    this->reset(make_Coord(size[1], 1), size, _device_backed);
  }

  /// Constructs a HostTensor given size, layout, and leading dimension
  HostMatrixRowMajor(TensorCoord const& size, Index ldm, bool _device_backed = true) {
    this->reset(make_Coord(ldm, 1), size, _device_backed);
  }

  /// Resizes a matrix
  void resize(MatrixCoord const &size, int ldm = 0, bool _device_backed = true) {
    this->reset(ldm, size, _device_backed);
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
