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
    \brief Host-side implementation of basic tensor operations.

    See cutlass/tensor_ref.h and cutlass/tensor_view.h for more details.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_view.h"
#include "tools/util/type_traits.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Data type of element stored within tensor
  typename Storage_,
  /// Rank of logical tensor
  int Rank_ = 4,
  /// Maps a Coord<Rank_> in the logical tensor index space to the internal n-D array
  typename MapFunc_ = IdentityTensorMapFunc<Rank_>,
  /// Rank of internal n-D array
  int StorageRank_ = Rank_,
  /// Index type used for coordinates
  typename Index_ = int,
  /// Index type used for offsets and pointer differences
  typename LongIndex_ = long long
>
class HostTensorView :
  public TensorView<Storage_, Rank_, MapFunc_, StorageRank_, Index_, LongIndex_> {
 public:
  /// Base class
  typedef TensorView<Storage_, Rank_, MapFunc_, StorageRank_, Index_, LongIndex_> Base;

  /// Storage type
  typedef typename Base::Storage Storage;

  /// Alias for underlying TensorRef_t
  typedef typename Base::TensorRef_t TensorRef_t;

  /// Index type
  typedef typename Base::Index Index;

  /// Coordinate in logical tensor space
  typedef typename TensorRef_t::TensorCoord TensorCoord;

  /// Coordinate in storage n-D array
  typedef typename TensorRef_t::StorageCoord StorageCoord;

  /// Stride vector in storage coordinate space
  /// Least significant stride is = 1 and not stored
  typedef typename TensorRef_t::StrideVector StrideVector;

  /// Long index type for pointer offsets
  typedef typename Base::LongIndex LongIndex;

  /// Rank of tensor index space
  static int const kRank = Base::kRank;

  //
  // Definitions included for backwards compatibility - These will be remmoved
  // in the next major release.
  //

  /// Base class
  typedef Base TensorView_t;

  //
  // These definitions are meaningful for rank=4 tensors.
  //

  /// Convention: depth is the first dimension
  static int const Dim_D = 0;

  /// Convention: height is the second dimension
  static int const Dim_H = 1;

  /// Convention: width is the third dimension
  static int const Dim_W = 2;

  /// Convention: channel is the second dimension
  static int const Dim_C = 3;

 public:

  //
  // Device and Host Methods
  //

  /// Default constructor
  HostTensorView() {}

  /// Helper to construct from pointer, stride, and size
  HostTensorView(
    Storage_ *_ptr,
    StrideVector const &_stride,
    TensorCoord const& _size
  ) : Base(TensorRef_t(_ptr, _stride), _size) {}

  /// Helper to construct from pointer, stride, and size
  HostTensorView(
    Storage_ *_ptr,
    StorageCoord const &_stride,
    TensorCoord const& _size
  ) : Base(TensorRef_t(_ptr, _stride), _size) {}

  /// Constructs a Tensor_view from a TensorRef_t and size assuming dense packing
  HostTensorView(
    TensorRef_t const& _ref,
    TensorCoord const& _size) : Base(_ref, _size) {}

  /// Assigns a tensor view
  HostTensorView& operator=(Base const& _tensor) {
    this->reset(_tensor.ref(), _tensor.size());
    return *this;
  }

  /// Returns a TensorView offset by a given amount
  CUTLASS_HOST_DEVICE
  HostTensorView operator+(TensorCoord const& b) const {
    HostTensorView result(*this);
    result.add_pointer_offset(this->offset(b));
    return result;
  }

  /// Returns a TensorRef_t offset by a given amount
  CUTLASS_HOST_DEVICE
  HostTensorView& operator+=(TensorCoord const& b) {
    this->add_pointer_offset(this->offset(b));
    return *this;
  }

  /// Returns a TensorRef_t offset by a given amount
  CUTLASS_HOST_DEVICE
  HostTensorView operator-(TensorCoord const& b) const {
    TensorRef_t result(*this);
    result.add_pointer_offset(-this->offset(b));
    return result;
  }

  /// Returns a TensorRef_t offset by a given amount
  CUTLASS_HOST_DEVICE
  HostTensorView& operator-=(TensorCoord const& b) {
    this->add_pointer_offset(-this->offset(b));
    return *this;
  }

  /// Recurses through all dimensions and applies a unary operation in place
  template <typename F>
  void elementwise_in_place(F& op, int dim = 0, TensorCoord const &start_coord = TensorCoord()) {

    TensorCoord coord(start_coord);
    for (int idx = 0; idx < this->size(dim); ++idx) {
      coord[dim] = idx;
      if (dim < kRank - 1) {
        elementwise_in_place(op, dim + 1, coord);
      } else {
        op(this->at(coord));
      }
    }
  }

  /// Recurses through all dimensions and applies a unary operator with no arguments
  template <typename F>
  void elementwise_stream(F& op, int dim = 0, TensorCoord const &start_coord = TensorCoord()) {

    TensorCoord coord(start_coord);
    for (int idx = 0; idx < this->size(dim); ++idx) {
      coord[dim] = idx;
      if (dim < kRank - 1) {
        elementwise_stream(op, dim + 1, coord);
      } else {
        this->at(coord) = op();
      }
    }
  }

  /// Recurses through all dimensions and applies a unary operator, supplying the logical
  /// coordinate within the tensor as an argument
  template <typename F>
  void elementwise_generate(F& op,
                            int dim = 0,
                            TensorCoord const & start_coord = TensorCoord()) {

    TensorCoord coord(start_coord);
    for (int idx = 0; idx < this->size(dim); ++idx) {
      coord[dim] = idx;
      if (dim < kRank - 1) {
        elementwise_generate(op, dim + 1, coord);
      } else {
        this->at(coord) = op(coord);
      }
    }
  }

  /// Recurses through all dimensions and applies a unary operator, supplying the logical
  /// coordinate within the tensor as an argument. Mutable.
  template <typename F>
  void elementwise_visit(F& op,
                         int dim = 0,
                         TensorCoord const & start_coord = TensorCoord()) const {

    TensorCoord coord(start_coord);
    for (int idx = 0; idx < this->size(dim); ++idx) {
      coord[dim] = idx;

      if (dim < kRank - 1) {
        elementwise_visit(op, dim + 1, coord);
      } else {
        op(this->at(coord), coord);
      }
    }
  }

  /// Recurses through all dimensions and applies a binary operation
  template <typename F, typename SrcTensorView>
  bool elementwise_in_place(F& op,
                            SrcTensorView const& tensor,
                            int dim = 0,
                            TensorCoord const &start_coord = TensorCoord()) {

    if (this->size(dim) != tensor.size(dim)) {
      return false;
    }

    TensorCoord coord(start_coord);
    for (int idx = 0; idx < this->size(dim); ++idx) {
      coord[dim] = idx;
      if (dim < kRank - 1) {
        elementwise_in_place(op, tensor, dim + 1, coord);
      } else {
        op(this->at(coord), tensor.at(coord));
      }
    }

    return true;
  }

  template <typename Src>
  struct LambdaBinaryAddition {
    void operator()(Storage_& a, Src b) const { a += Storage_(b); }
  };

  template <typename Src>
  struct LambdaBinarySubtraction {
    void operator()(Storage_& a, Src b) const { a -= Storage_(b); }
  };

  template <typename Src>
  struct LambdaBinaryMultiplication {
    void operator()(Storage_& a, Src b) const { a *= Storage_(b); }
  };

  template <typename Src>
  struct LambdaBinaryDivision {
    void operator()(Storage_& a, Src b) const { a /= Storage_(b); }
  };

  /// Accumulate in place
  template <typename SrcTensorView>
  HostTensorView& operator+=(SrcTensorView const& tensor) {
    LambdaBinaryAddition<typename SrcTensorView::Storage> op;
    elementwise_in_place(op, tensor);

    return *this;
  }

  /// Subtract in place
  template <typename SrcTensorView>
  HostTensorView& operator-=(SrcTensorView const& tensor) {
    LambdaBinarySubtraction<typename SrcTensorView::Storage> op;
    elementwise_in_place(op, tensor);

    return *this;
  }

  /// Multiply in place
  template <typename SrcTensorView>
  HostTensorView& operator*=(SrcTensorView const& tensor) {
    LambdaBinaryMultiplication<typename SrcTensorView::Storage> op;
    elementwise_in_place(op, tensor);

    return *this;
  }

  /// Divide in place
  template <typename SrcTensorView>
  HostTensorView& operator/=(SrcTensorView const& tensor) {
    LambdaBinaryDivision<typename SrcTensorView::Storage> op;
    elementwise_in_place(op, tensor);

    return *this;
  }

  /// Comparison operator
  struct EqualsOperator {
    bool equal;
    Storage_ eps;

    EqualsOperator(Storage_ _epsilon) : equal(true), eps(_epsilon) {}

    void operator()(Storage_ a, Storage_ b) {
      if (std::abs(Storage_(a - b)) > eps * std::max(std::abs(a), std::abs(b))) {
        equal = false;
      }
    }
  };

  /// equality with epsilon tolerance
  bool equals(Base const& tensor, Storage epsilon) const {
    EqualsOperator comparison_op(epsilon);
    bool equal_size = elementwise_in_place(comparison_op, tensor);

    return equal_size && comparison_op.equal;
  }

  /// Compares two values which are smaller or equal to a long long int
  struct BitEqualsOperator {
    bool equal;
    long long eps;
    uint64_t index;

    BitEqualsOperator(long long _ulps_threshold) : equal(true), eps(_ulps_threshold), index(0) {}

    void operator()(Storage_ a, Storage_ b) {
      // convert bits to integers
      long long bits_a = 0;
      long long bits_b = 0;

      *reinterpret_cast<Storage_*>(&bits_a) = TypeTraits<Storage_>::remove_negative_zero(a);
      *reinterpret_cast<Storage_*>(&bits_b) = TypeTraits<Storage_>::remove_negative_zero(b);

      // compute diff
      long long ulps = bits_a - bits_b;
      if (std::abs(ulps) > eps) {
        equal = false;
      }
      index++;
    }
  };

  /// equality with ulps tolerance
  bool bit_equals(Base const& tensor, long long ulps_threshold = 0) {
    BitEqualsOperator comparison_op(ulps_threshold);
    bool equal_size = elementwise_in_place(comparison_op, tensor);

    return equal_size && comparison_op.equal;
  }

  /// Fills with random data
  template <typename Gen>
  void fill_random(Gen generator) {
    elementwise_stream(generator);
  }

  /// Procedurally assigns elements
  template <typename Gen>
  void generate(Gen generator) {
    elementwise_generate(generator);
  }

  /// Procedurally visits elements
  template <typename Gen>
  void visit(Gen& generator) const {
    elementwise_visit(generator);
  }

  /// Generator to fill a tensor with the identity matrix
  struct LambdaFillIdentity {
    Storage_ operator()(TensorCoord const& coord) {
      return (coord.at(1) == coord.at(2) ? Storage_(1) : Storage_(0));
    }
  };

  /// initializes with identity
  void fill_identity() {
    LambdaFillIdentity op;
    elementwise_generate(op);
  }

  /// Lambda for fill_linear()
  struct LambdaFillLinear {
    TensorCoord v_;
    Storage_ offset_;

    LambdaFillLinear(TensorCoord const& _v, Storage_ _offset) : v_(_v), offset_(_offset) {}

    Storage_ operator()(TensorCoord const& coord) {
      return Storage_(v_.template dot<int>(coord)) + offset_;
    }
  };

  /// computes elements as a linear combination of their coordinates
  void fill_linear(TensorCoord v, Storage_ offset = Storage_(0)) {
    LambdaFillLinear lambda(v, offset);
    elementwise_generate(lambda);
  }

  /// computes elements as a linear combination of their coordinates
  void fill_sequential(Storage_ v = Storage_(1), Storage_ offset = Storage_(0)) {
    int const count = this->size().count();
    for (int i = 0; i < count; ++i) {
      this->data()[i] = Storage_(i);
    }
  }

  /// Returns a constant value
  struct LambdaFillValue {
    Storage_ value;

    LambdaFillValue(Storage_ _value) : value(_value) {}

    Storage_ operator()() { return value; }
  };

  /// fills with a value
  void fill(Storage_ val = Storage_(0)) {
    LambdaFillValue op(val);
    elementwise_stream(op);
  }

  /// Conversion from Src to T
  template <typename Src>
  struct LambdaAssign {
    void operator()(Storage_& a, Src b) const { a = Storage_(b); }
  };

  /// copies from external data source and performs type conversion
  template <
    typename SrcType,
    typename SrcMapFunc_,
    int SrcStorageRank_,
    typename SrcIndex_,
    typename SrcLongIndex_
  >
  void fill(
    TensorView<SrcType, kRank, SrcMapFunc_, SrcStorageRank_, SrcIndex_, SrcLongIndex_> const& tensor) {

    LambdaAssign<SrcType> op;
    elementwise_in_place(op, tensor);
  }

  /// Computes a norm
  struct LambdaNorm {
    double sum;

    LambdaNorm() : sum(0) {}

    void operator()(Storage const& element) {
      double value(element);
      double conj(element);  

      sum += value * conj;
    }
  };

  /// Computes the norm of the matrix in double-precision
  double norm() const {
    LambdaNorm op;
    elementwise_in_place(op);

    return std::sqrt(op.sum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

