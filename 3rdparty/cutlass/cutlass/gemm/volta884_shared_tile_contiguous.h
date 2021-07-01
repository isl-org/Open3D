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
    \brief Defines structural properties for GEMM targeting Volta's mma.sync instruction
*/

#pragma once


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Congruous loading
//

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Store iterator specialized for A.column_major
template <
    /// Specifies threadblock tile shape
    typename Tile_,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    int WarpDelta>
struct Volta884ThreadblockMultiplicandStoreIterator<GemmOperand::kA,
                                                    MatrixLayout::kColumnMajor,
                                                    Tile_,
                                                    WarpCount,
                                                    WarpDelta> {
  //
  // Constant and type definitions
  //

  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kA;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// Shape of thread-block multiplicand
  typedef Tile_ Tile;

  /// Number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp accumulator tiles along the outer dimension
  static int const kWarpDelta = WarpDelta;

  /// This implementation is specialized for 128b loads
  static int const kAccessSize = 8;

  /// Swizzled store iterator
  struct ThreadOffset {
    __device__ Coord<4> operator()() const {
      int warp_id = (threadIdx.x >> 5);
      int lane_id = (threadIdx.x & 0x1f);

      int k_idx = warp_id;

      // This is an 8-element vector within one 32x32 tile
      int vec_idx = lane_id & 3;
      int vec_col = (vec_idx / 2);

      int t4t3 = (lane_id >> 3);
      int col_rotate = ((lane_id >> 1) & 2) | (lane_id & 1);

      int t_col = (vec_col << 2) | (col_rotate ^ t4t3);

      Coord<4> offset = make_Coord(k_idx, col_rotate, t_col, 0);

      return offset;
    }
  };

  /// Projects the threadblock tile
  typedef typename GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Stored tile has a structure designed for efficient MIO storing and loading
  typedef Shape<(OperandShape::kH >> 2),  // one 3D tile per four elements in the K dimension
                (OperandShape::kW >> 4),  // four rows of SMEM per 64xK tile
                kAccessSize,              // Eight banks of MIO
                kAccessSize>
      VectorizedShape;  // 128b stores

  /// Offset between stores
  typedef Shape<WarpCount, 1, 1, 1> Delta;

  /// Number of iterations
  typedef Shape<(VectorizedShape::kD / WarpCount), (OperandShape::kW >> 6), 1, 1> Iterations;

  /// Source tile traits
  typedef TileTraits<VectorizedShape, Delta, Iterations, ThreadOffset, kAccessSize> Traits;

  /// Scalar type
  typedef half Scalar;

  /// Index type
  typedef int Index;

  /// Index type
  typedef int LongIndex;

  //
  // Derived types
  //

  /// Tensor reference
  typedef TensorRef<Scalar, 4> TensorRef;

  /// Predicate vector
  typedef PredicateVector<ShapeCount<Iterations>::kCount> PredicateVector;

  /// Fragment definition
  typedef Fragment<Scalar, ShapeCount<Iterations>::kCount * kAccessSize> Fragment;

  /// Elements loaded by one instruction
  typedef typename Vectorize<Scalar, kAccessSize>::Type AccessType;

  /// The fragment iterator.
  typedef FragmentIterator<Fragment, Iterations, AccessType> FragmentIterator;

  /// The fragment const iterator.
  typedef FragmentConstIterator<Fragment, Iterations, AccessType> FragmentConstIterator;

  /// Strides into expected SMEM tile
  typedef typename ShapeStrides<VectorizedShape, 1>::Shape Strides;

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Parameters object
  struct Params {
    //
    // Data members
    //

    /// Pointer to element type
    Scalar *pointer;

    /// Strides
    Coord<4> stride;

    //
    // Methods
    //

    /// Constructs a parameters object
    CUTLASS_HOST_DEVICE
    Params(Scalar *_pointer = 0)
        : pointer(_pointer),
          stride(make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC)) {}

    /// Constructs a params object from a TensorRef
    CUTLASS_HOST_DEVICE
    Params(TensorRef const &ref): pointer(ref.data()), stride(make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC)) { }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Constructs a store iterator
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator(
      Params const &_params,
      Coord<4> const &_block_offset = make_Coord(0, 0, 0, 0),
      ThreadOffset offset_func = ThreadOffset())
      : params(_params) {
    // Compute initial thread offset
    Coord<4> offset = offset_func();

    params.pointer += (_block_offset + offset).template dot<int>(params.stride);
  }

  /// Stores a fragment
  CUTLASS_DEVICE void store(Fragment const &fragment,
                            Coord<4> const &offset = make_Coord(0, 0, 0, 0)) const {
    FragmentConstIterator frag_iterator(fragment);

    // Iterate over each store
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          int idx = w + Iterations::kW * h;

          int row = idx * 4;

          Coord<4> sts_offset =
              make_Coord(d, row, 0, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kC);

          Store<typename Fragment::Element, VectorizedShape::kC, kMemorySpace>::store(
              reinterpret_cast<AccessType const &>(frag_iterator.at(d, h, w, 0)),
              params.pointer,
              params.stride.template dot<int>(sts_offset + offset));
        }
      }
    }
  }

  /// Increments store iterator to next tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &increment(int count = 1) {
    params.pointer +=
        make_Coord(VectorizedShape::kD * count, 0, 0, 0).template dot<int>(params.stride);
    return *this;
  }

  /// Increments to next tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &operator++() { return increment(); }

  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &operator+=(int count) {
    return increment(count);
  }

  /// Increments store iterator to previous tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &decrement(int count = 1) {
    params.pointer -=
        make_Coord(VectorizedShape::kD * count, 0, 0, 0).template dot<int>(params.stride);
    return *this;
  }

  /// Increments to subsequent tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &operator--() { return decrement(); }

  /// Decrements to previous tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &operator-=(int count) {
    return decrement(count);
  }

  /// Stores a fragment and increments in the K dimension
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &store_post_increment(
      Fragment const &fragment, Coord<4> const &offset = make_Coord(0, 0, 0, 0)) {
    store(fragment, offset);
    return increment();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator to load a fragment for each warp-level tile specialized for A.column_major
template <
    /// Specifies threadblock tile shape
    typename Tile_,
    /// Specifies the warp tile shape
    typename WarpTile_,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    typename WarpDelta_>
struct Volta884WarpMultiplicandLoadIterator<GemmOperand::kA,
                                            MatrixLayout::kColumnMajor,
                                            Tile_,
                                            WarpTile_,
                                            WarpCount,
                                            WarpDelta_> {
  //
  // Constant and type definitions
  //

  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kA;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// Shape of thread-block multiplicand
  typedef Tile_ Tile;

  /// Shape of warp-tile matrix operation
  typedef WarpTile_ WarpTile;

  /// Hard-coded tile shape
  typedef Shape<4, 32, 32> InterleavedTileShape;

  /// Number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp accumulator tiles along the outer dimension
  typedef WarpDelta_ WarpDelta;

  /// Two SMEM read pointers are needed
  static int const kPointerCount = (WarpDelta::kW == 1 ? 2 : 1);

  /// This implementation is specialized for 128b loads
  static int const kAccessSize = 8;

  /// Swizzled store iterator
  struct ThreadOffset {
    /// Compute thread offset coordinate for each pointer
    CUTLASS_DEVICE Coord<4> operator()(int pointer_idx = 0) const {
      // Determine the warp's reading location within the SMEM tile
      int warp_id = ((threadIdx.x >> 5) % WarpDelta::kW);

      // This is an 8-element vector within one 32x32 tile
      int lane_id = (threadIdx.x & 0x1f);
      int vec_row = (lane_id >> 4);
      int vec_col = ((lane_id & 4) >> 2);

      int tile_row = pointer_idx * 2 + vec_row;

      // Column rotation function
      int t_col = (vec_col * 4);
      if (pointer_idx == 1 || (WarpDelta::kW > 1 && (warp_id & 1))) {
        vec_row |= 2;
      }

      t_col = t_col | ((lane_id & 3) ^ vec_row);

      Coord<4> offset = make_Coord(0, warp_id * 2 + tile_row, t_col, 0);

      return offset;
    }
  };

  /// Projects the threadblock tile
  typedef typename GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Stored tile has a structure designed for efficient MIO storing and loading
  typedef Shape<(OperandShape::kH >> 2),  // one 3D tile per four elements in the K dimension
                (OperandShape::kW >> 4),  // four rows of SMEM per 64xK tile
                kAccessSize,              // Eight banks of MIO
                kAccessSize>
      VectorizedShape;  // 128b stores

  /// Offset between acceses
  typedef typename platform::conditional<WarpDelta::kW == 1,
                                         Shape<1, 0, 0, 0>,
                                         Shape<1, 2 * WarpDelta::kW, 0, 0> >::type Delta;

  /// Number of iterations
  typedef Shape<1, WarpTile::kW / InterleavedTileShape::kW, 1, 1> Iterations;

  /// Source tile traits
  typedef TileTraits<VectorizedShape, Delta, Iterations, ThreadOffset, kAccessSize> Traits;

  /// Scalar type
  typedef half Scalar;

  /// Index type
  typedef int Index;

  /// Index type
  typedef int LongIndex;

  //
  // Derived types
  //

  /// Tensor reference
  typedef TensorRef<Scalar, 4> TensorRef;

  /// Predicate vector
  typedef PredicateVector<ShapeCount<Iterations>::kCount> PredicateVector;

  /// Fragment definition
  typedef Fragment<Scalar, ShapeCount<Iterations>::kCount * kAccessSize> Fragment;

  /// Elements loaded by one instruction
  typedef typename Vectorize<Scalar, kAccessSize>::Type AccessType;

  /// The fragment iterator.
  typedef FragmentIterator<Fragment, Iterations, AccessType> FragmentIterator;

  /// The fragment const iterator.
  typedef FragmentConstIterator<Fragment, Iterations, AccessType> FragmentConstIterator;

  /// Strides into expected SMEM tile
  typedef typename ShapeStrides<VectorizedShape, kAccessSize>::Shape Strides;

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Parameters object
  struct Params {
    //
    // Data members
    //

    /// Base pointer to SMEM allocation
    Scalar const *pointer;

    /// SMEM strides
    Coord<4> stride;

    //
    // Methods
    //

    /// Constructs a parameters object
    CUTLASS_HOST_DEVICE
    Params(Scalar const *_pointer = 0)
        : pointer(_pointer),
          stride(make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC)) {}

    /// Constructs a params object from a TensorRef
    CUTLASS_HOST_DEVICE
    Params(TensorRef const &ref): pointer(ref.data()), stride(make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC)) { }
  };

  //
  // Data members
  //

  // A.column requires two SMEM pointers.
  // Because Params only supplies a base pointer and strides, there is no usual params
  // data member. Instead, it is used to initialize the following.

  /// Pointer to SMEM allocation.
  Scalar const *pointer[kPointerCount];

  /// SMEM strides
  Coord<4> stride;

  //
  // Methods
  //

  /// Constructs a load iterator
  CUTLASS_DEVICE Volta884WarpMultiplicandLoadIterator(
      Params const &_params,
      Coord<4> const &_block_offset = make_Coord(0, 0, 0, 0),
      ThreadOffset offset_func = ThreadOffset())
      : stride(_params.stride) {
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < kPointerCount; ++idx) {
      Coord<4> offset = offset_func(idx);

      pointer[idx] = _params.pointer + (_block_offset + offset).template dot<int>(stride);
    }
  }

  /// Loads a fragment
  CUTLASS_DEVICE void load(Fragment &fragment,
                           Coord<4> const &offset = make_Coord(0, 0, 0, 0)) const {
    FragmentIterator frag_iterator(fragment);

    // Iterate over each load
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          // Pointers mapped to Iterations::kH dimension
          Scalar const *_pointer = pointer[(kPointerCount == 2 ? h : 0)];

          Coord<4> lds_offset =
              make_Coord(d, h, w, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kC);

          Load<typename Fragment::Element, VectorizedShape::kC, kMemorySpace>::load(
              reinterpret_cast<AccessType &>(frag_iterator.at(d, h, w, 0)),
              _pointer,
              stride.template dot<int>(lds_offset + offset));
        }
      }
    }
  }

  /// Loads a fragment and increments to next K-index
  CUTLASS_DEVICE void load_post_increment(Fragment &fragment,
                                          Coord<4> const &offset = make_Coord(0, 0, 0, 0)) {
    load(fragment, offset);

    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      pointer[ptr_idx] += make_Coord(1, 0, 0, 0).template dot<int>(stride);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Store iterator specialized for B.row_major
template <
    /// Specifies threadblock tile shape
    typename Tile_,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    int WarpDelta>
struct Volta884ThreadblockMultiplicandStoreIterator<GemmOperand::kB,
                                                    MatrixLayout::kRowMajor,
                                                    Tile_,
                                                    WarpCount,
                                                    WarpDelta> {
  //
  // Constant and type definitions
  //

  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kB;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// Shape of thread-block multiplicand
  typedef Tile_ Tile;

  /// Number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp accumulator tiles along the outer dimension
  static int const kWarpDelta = WarpDelta;

  /// This implementation is specialized for 128b loads
  static int const kAccessSize = 8;

  /// Index type
  typedef int Index;

  /// Index type
  typedef int LongIndex;

  /// Swizzled store iterator
  struct ThreadOffset {
    CUTLASS_DEVICE Coord<4> operator()() const {
      int warp_id = (threadIdx.x >> 5);
      int lane_id = (threadIdx.x & 0x1f);

      int k_idx = warp_id;

      // This is an 8-element vector within one 32x32 tile
      int vec_idx = lane_id & 3;
      int vec_col = (vec_idx / 2);

      int t4t3 = (lane_id >> 3);
      int col_rotate = ((lane_id >> 1) & 2) | (lane_id & 1);

      int t_col = (vec_col << 2) | (col_rotate ^ t4t3);

      Coord<4> offset = make_Coord(k_idx, col_rotate , t_col, 0);

      return offset;
    }
  };

  /// Projects the threadblock tile
  typedef typename GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Stored tile has a structure designed for efficient MIO storing and loading
  typedef Shape<(OperandShape::kH >> 2),  // one 3D tile per four elements in the K dimension
                (OperandShape::kW >> 4),  // four rows of SMEM per 64xK tile
                kAccessSize,              // Eight banks of MIO
                kAccessSize>
      VectorizedShape;  // 128b stores

  /// Offset between stores
  typedef Shape<WarpCount, 1, 1, 1> Delta;

  /// Number of iterations
  typedef Shape<(VectorizedShape::kD / WarpCount), (OperandShape::kW >> 6), 1, 1> Iterations;

  /// Source tile traits
  typedef TileTraits<VectorizedShape, Delta, Iterations, ThreadOffset, kAccessSize> Traits;

  /// Scalar type
  typedef half Scalar;

  //
  // Derived types
  //

  /// Tensor reference
  typedef TensorRef<Scalar, 4> TensorRef;

  /// Predicate vector
  typedef PredicateVector<ShapeCount<Iterations>::kCount> PredicateVector;

  /// Fragment definition
  typedef Fragment<Scalar, ShapeCount<Iterations>::kCount * kAccessSize> Fragment;

  /// Elements loaded by one instruction
  typedef typename Vectorize<Scalar, kAccessSize>::Type AccessType;

  /// The fragment iterator.
  typedef FragmentIterator<Fragment, Iterations, AccessType> FragmentIterator;

  /// The fragment const iterator.
  typedef FragmentConstIterator<Fragment, Iterations, AccessType> FragmentConstIterator;

  /// Strides into expected SMEM tile
  typedef typename ShapeStrides<VectorizedShape, 1>::Shape Strides;

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Parameters object
  struct Params {
    //
    // Data members
    //

    /// Pointer to element type
    Scalar *pointer;

    /// Strides
    Coord<4> stride;

    //
    // Methods
    //

    /// Constructs a parameters object
    CUTLASS_HOST_DEVICE
    Params(Scalar *_pointer = 0)
        : pointer(_pointer),
          stride(make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC)) {}

    /// Constructs a params object from a TensorRef
    CUTLASS_HOST_DEVICE
    Params(TensorRef const &ref): pointer(ref.data()), stride(make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC)) { }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Constructs a store iterator
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator(
      Params const &_params,
      Coord<4> const &_block_offset = make_Coord(0, 0, 0, 0),
      ThreadOffset offset_func = ThreadOffset())
      : params(_params) {
    // Compute initial offset for each thread
    Coord<4> offset = offset_func();

    params.pointer += (_block_offset + offset).template dot<int>(params.stride);
  }

  /// Stores a fragment
  CUTLASS_DEVICE void store(Fragment const &fragment,
                            Coord<4> const &offset = make_Coord(0, 0, 0, 0)) const {
    FragmentConstIterator frag_iterator(fragment);

    // Iterate over each store
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          int idx = w + Iterations::kW * h;
          int row = idx * 4;

          Coord<4> sts_offset =
              make_Coord(d, row, 0, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kC);

          Index _offset = params.stride.template dot<int>(sts_offset + offset);

          Store<typename Fragment::Element, VectorizedShape::kC, kMemorySpace>::store(
              reinterpret_cast<AccessType const &>(frag_iterator.at(d, h, w, 0)),
              params.pointer,
              _offset);
        }
      }
    }
  }

  /// Increments store iterator to next tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &increment(int count = 1) {
    params.pointer +=
        make_Coord(VectorizedShape::kD * count, 0, 0, 0).template dot<int>(params.stride);
    return *this;
  }

  /// Increments to next tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &operator++() { return increment(); }

  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &operator+=(int count) {
    return increment(count);
  }

  /// Increments store iterator to previous tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &decrement(int count = 1) {
    params.pointer -=
        make_Coord(VectorizedShape::kD * count, 0, 0, 0).template dot<int>(params.stride);
    return *this;
  }

  /// Increments to subsequent tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &operator--() { return decrement(); }

  /// Decrements to previous tile
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &operator-=(int count) {
    return decrement(count);
  }

  /// Stores a fragment and increments in the K dimension
  CUTLASS_DEVICE Volta884ThreadblockMultiplicandStoreIterator &store_post_increment(
      Fragment const &fragment, Coord<4> const &offset = make_Coord(0, 0, 0, 0)) {
    store(fragment, offset);
    return increment();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator to load a fragment for each warp-level tile specialized for B.row_major
template <
    /// Specifies threadblock tile shape
    typename Tile_,
    /// Specifies the warp tile shape
    typename WarpTile_,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    typename WarpDelta_>
struct Volta884WarpMultiplicandLoadIterator<GemmOperand::kB,
                                            MatrixLayout::kRowMajor,
                                            Tile_,
                                            WarpTile_,
                                            WarpCount,
                                            WarpDelta_> {
  //
  // Constant and type definitions
  //

  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kB;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// Shape of thread-block multiplicand
  typedef Tile_ Tile;

  /// Shape of warp-tile matrix operation
  typedef WarpTile_ WarpTile;

  /// Hard-coded tile shape
  typedef Shape<4, 32, 32> InterleavedTileShape;

  /// Number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp accumulator tiles along the outer dimension
  typedef WarpDelta_ WarpDelta;

  /// This implementation is specialized for 128b loads
  static int const kAccessSize = 8;

  /// Swizzled store iterator
  struct ThreadOffset {
    /// Computes the initial offset
    CUTLASS_DEVICE Coord<4> operator()(int pointer_idx) const {
      // Determine the warp's reading location within the SMEM tile
      int warp_id = ((threadIdx.x >> 5) / WarpDelta::kW);

      // This is an 8-element vector within one 32x32 tile
      int lane_id = (threadIdx.x & 0x1f);
      int vec_row = (lane_id >> 4);
      int vec_col = ((lane_id & 8) >> 3);

      int tile_row = pointer_idx * 2 + vec_row;

      // Column rotation function
      int t_col = (vec_col * 4);
      if (pointer_idx == 1 || (WarpDelta::kH > 1 && (warp_id & 1))) {
        vec_row |= 2;
      }

      t_col = t_col | ((lane_id & 3) ^ vec_row);
      Coord<4> offset = make_Coord(0, warp_id * 2 + tile_row, t_col, 0);

      return offset;
    }
  };

  /// Projects the threadblock tile
  typedef typename GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Stored tile has a structure designed for efficient MIO storing and loading
  typedef Shape<(OperandShape::kH >> 2),  // one 3D tile per four elements in the K dimension
                (OperandShape::kW >> 4),  // four rows of SMEM per 64xK tile
                kAccessSize,              // Eight banks of MIO
                kAccessSize>
      VectorizedShape;  // 128b stores

  /// Delta between accesses
  typedef typename platform::conditional<WarpDelta::kH == 1,
                                         Shape<1, 0, 0, 0>,
                                         Shape<1, 2 * WarpDelta::kH, 0, 0> >::type Delta;

  /// Number of iterations
  typedef Shape<1, WarpTile::kH / InterleavedTileShape::kH, 1, 1> Iterations;

  /// Source tile traits
  typedef TileTraits<VectorizedShape, Delta, Iterations, ThreadOffset, kAccessSize> Traits;

  /// Scalar type
  typedef half Scalar;

  /// Index type
  typedef int Index;

  /// Index type
  typedef int LongIndex;

  //
  // Derived types
  //

  /// Tensor reference
  typedef TensorRef<Scalar, 4> TensorRef;

  /// Predicate vector
  typedef PredicateVector<ShapeCount<Iterations>::kCount> PredicateVector;

  /// Fragment definition
  typedef Fragment<Scalar, ShapeCount<Iterations>::kCount * kAccessSize> Fragment;

  /// Elements loaded by one instruction
  typedef typename Vectorize<Scalar, kAccessSize>::Type AccessType;

  /// The fragment iterator.
  typedef FragmentIterator<Fragment, Iterations, AccessType> FragmentIterator;

  /// The fragment const iterator.
  typedef FragmentConstIterator<Fragment, Iterations, AccessType> FragmentConstIterator;

  /// Strides into expected SMEM tile
  typedef typename ShapeStrides<VectorizedShape, 1>::Shape Strides;

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Number of SMEM read pointers needed
  static int const kPointerCount = (WarpDelta::kH == 1 ? 2 : 1);

  /// Parameters object
  struct Params {
    //
    // Data members
    //

    /// Pointer to element type
    Scalar const *pointer;

    /// Strides
    Coord<4> stride;

    //
    // Methods
    //

    /// Constructs a parameters object
    CUTLASS_HOST_DEVICE
    Params(Scalar const *_pointer = 0)
        : pointer(_pointer),
          stride(make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC)) {}

    /// Constructs a params object from a TensorRef
    CUTLASS_HOST_DEVICE
    Params(TensorRef const &ref): pointer(ref.data()), stride(make_Coord(Strides::kD, Strides::kH, Strides::kW, Strides::kC)) { }
  };

  //
  // Data members
  //

  /// Pointer to element type
  Scalar const *pointer[kPointerCount];

  /// Strides
  Coord<4> stride;

  //
  // Methods
  //

  /// Constructs a load iterator
  CUTLASS_DEVICE Volta884WarpMultiplicandLoadIterator(
      Params const &_params,
      Coord<4> const &_block_offset = make_Coord(0, 0, 0, 0),
      ThreadOffset offset_func = ThreadOffset())
      : stride(_params.stride) {
    CUTLASS_PRAGMA_UNROLL
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      Coord<4> offset = offset_func(ptr_idx);

      pointer[ptr_idx] = _params.pointer + (_block_offset + offset).template dot<int>(stride);
    }
  }

  /// Stores a fragment
  CUTLASS_DEVICE void load(Fragment &fragment,
                           Coord<4> const &offset = make_Coord(0, 0, 0, 0)) const {
    FragmentIterator frag_iterator(fragment);

    // Iterate over each load
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          // Pointers mapped to Iterations::kH dimension
          Scalar const *_pointer = pointer[(kPointerCount == 2 ? h : 0)];

          Coord<4> lds_offset =
              make_Coord(d, h, w, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kC);

          Load<typename Fragment::Element, VectorizedShape::kC, kMemorySpace>::load(
              reinterpret_cast<AccessType &>(frag_iterator.at(d, h, w, 0)),
              _pointer,
              stride.template dot<int>(lds_offset + offset));
        }
      }
    }
  }

  /// Loads a fragment and increments to next K-index
  CUTLASS_DEVICE void load_post_increment(Fragment &fragment,
                                          Coord<4> const &offset = make_Coord(0, 0, 0, 0)) {
    load(fragment, offset);

    CUTLASS_PRAGMA_UNROLL
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      pointer[ptr_idx] += make_Coord(1, 0, 0, 0).template dot<int>(stride);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

