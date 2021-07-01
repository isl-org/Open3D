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
// Crosswise loading
//

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Store iterator specialized for A.row_major
template <
    /// Specifies threadblock tile shape
    typename Tile_,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    int WarpDelta>
struct Volta884ThreadblockMultiplicandStoreIterator<GemmOperand::kA,
                                                    MatrixLayout::kRowMajor,
                                                    Tile_,
                                                    WarpCount,
                                                    WarpDelta> {
  //
  // Assertions
  //

  // Crosswise loaders may only span 32 x 128b along the K dimension
  static_assert(!(Tile_::kW % 8) && (Tile_::kW <= 256),
                "Tile dimensions must be divisible by 8 elements, and the K dimension may not span "
                "more than what a single warp can load");

  //
  // Constant and type definitions
  //

  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kA;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// Shape of thread-block multiplicand
  typedef Tile_ Tile;

  /// Number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp accumulator tiles along the outer dimension
  static int const kWarpDelta = WarpDelta;

  /// LDG.128 loads
  static int const kLdgAccessSize = 8;

  /// This implementation is specialized for 64b loads
  static int const kAccessSize = 4;

  /// Projects the threadblock tile
  typedef typename GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Stored tile has a structure designed for efficient MIO storing and loading
  typedef Shape<(OperandShape::kW >> 2),  // one 3D tile per four elements in the K dimension
                (OperandShape::kH >> 4),  // four rows of SMEM per 64xK tile
                16,                       // Sixteen banks of MIO
                kAccessSize>
      VectorizedShape;

  /// Offset between stores
  typedef Shape<WarpCount, 1, 1, 1> Delta;

  /// Shape of tile
  typedef Shape<1, 8, 4> WarpStoreCoverage;

  /// Number of iterations
  typedef Shape<
      // # of LDG.128s along the strided (outer) dimension
      OperandShape::kH / (WarpStoreCoverage::kH * kWarpCount),
      // # of LDG.128s along the contiguous (K) dimension
      OperandShape::kW / (WarpStoreCoverage::kW * kLdgAccessSize),
      // # STSs per LDG
      (kLdgAccessSize / kAccessSize),
      1>
      Iterations;

  /// Swizzled store iterator
  struct ThreadOffset {
    __device__ Coord<4> operator()(int ptr_idx) const {
      int warp_id = (threadIdx.x >> 5);
      int lane_id = (threadIdx.x & 0x1f);

      // Assumes a contiguous/blocked warp loading strategy
      int load_tile_idx = warp_id * Iterations::kD;

      // Compute swizzled destination address
      int lane_w = (lane_id % WarpStoreCoverage::kW);
      int store_k_idx = lane_w * 2;

      int dest_tile_idx = load_tile_idx / 4;

      int dest_row = ((load_tile_idx >> 1) & 1);
      int dest_bank = (lane_id & 0x0f) ^ ((lane_id >> 4) & 1) ^ (ptr_idx << 1);

      Coord<4> offset = make_Coord(store_k_idx, dest_tile_idx * 2 + dest_row, dest_bank, 0);

      return offset;
    }
  };

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

  /// Store iterators require two pointers
  static int const kPointerCount = 2;

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

  /// Pointer to element type
  Scalar *pointer[kPointerCount];

  /// Strides
  Coord<4> stride;

  //
  // Methods
  //

  /// Constructs a store iterator
  __device__ Volta884ThreadblockMultiplicandStoreIterator(
      Params const &_params,
      Coord<4> const &_block_offset = make_Coord(0, 0, 0, 0),
      ThreadOffset offset_func = ThreadOffset())
      : stride(_params.stride) {
    // Initialize each pointer
    CUTLASS_PRAGMA_UNROLL
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      Coord<4> offset = offset_func(ptr_idx);
      pointer[ptr_idx] = _params.pointer + (_block_offset + offset).template dot<int>(stride);
    }
  
    if (((threadIdx.x >> 5) * Iterations::kD) & 2) {
      Scalar *tmp = pointer[0];
      pointer[0] = pointer[1];
      pointer[1] = tmp;
    }
  }

  /// Stores a fragment
  __device__ void store(Fragment const &fragment,
                        Coord<4> const &offset = make_Coord(0, 0, 0, 0)) const {
    FragmentConstIterator frag_iterator(fragment);

    // Iterate over each store
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {  // strided LDG.128s

      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {  // contiguous LDG.128s

        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {  // 2x STS operations per LDG

          int k_idx = w + h * 8;
          int smem_row = (d >> 1);

          // Two store pointers
          Scalar *_pointer = pointer[(d & 1) ^ ((d >> 1) & 1)];
          
          Coord<4> sts_offset = make_Coord(k_idx, smem_row, 0, 0);

          Store<typename Fragment::Element, kAccessSize, kMemorySpace>::store(
              reinterpret_cast<AccessType const &>(frag_iterator.at(d, h, w, 0)),
              _pointer,
              stride.template dot<int>(sts_offset + offset));
        }
      }
    }
  }

  /// Increments store iterator to next tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &increment(int count = 1) {
    CUTLASS_PRAGMA_UNROLL
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      pointer[ptr_idx] +=
          make_Coord(VectorizedShape::kD * count, 0, 0, 0).template dot<int>(stride);
    }
    return *this;
  }

  /// Increments to next tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &operator++() { return increment(1); }

  __device__ Volta884ThreadblockMultiplicandStoreIterator &operator+=(int count) {
    return increment(count);
  }

  /// Increments store iterator to previous tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &decrement(int count = 1) {
    CUTLASS_PRAGMA_UNROLL 
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      pointer[ptr_idx] -=
          make_Coord(VectorizedShape::kD * count, 0, 0, 0).template dot<int>(stride);
    }
    return *this;
  }

  /// Increments to subsequent tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &operator--() { return decrement(1); }

  /// Decrements to previous tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &operator-=(int count) {
    return decrement(count);
  }

  /// Stores a fragment and increments in the K dimension
  __device__ Volta884ThreadblockMultiplicandStoreIterator &store_post_increment(
      Fragment const &fragment, Coord<4> const &offset = make_Coord(0, 0, 0, 0)) {
    store(fragment, offset);
    return increment();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator to load a fragment for each warp-level tile specialized for A.row_major
template <
    /// Specifies threadblock tile shape
    typename Tile_,
    /// Specifies the shape of thewarp tile
    typename WarpTile_,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    typename WarpDelta_>
struct Volta884WarpMultiplicandLoadIterator<GemmOperand::kA,
                                            MatrixLayout::kRowMajor,
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
    /// Compute thread offset coordinate for each pointer
    __device__ Coord<4> operator()(int ptr_idx) const {
      int warp_id = ((threadIdx.x >> 5) % WarpDelta::kW);
      int lane_id = (threadIdx.x & 0x1f);

      int lane_in_quad = (lane_id & 0x3);
      int quad_id = ((lane_id >> 2) & 0x7);

      int oct_row_id = ((quad_id >> 1) & 2) | (quad_id & 1);
      int oct_row = (oct_row_id & 1);
      int oct_left_right = (oct_row_id & 1) ^ ((oct_row_id >> 1) & 1) ^ ptr_idx;

      Coord<4> offset = make_Coord(0, warp_id * 2 + oct_row, lane_in_quad * 2 + oct_left_right, 0);

      return offset;
    }
  };

  /// Projects the threadblock tile
  typedef typename GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Loaded tile has a structure designed for efficient MIO storing and loading
  typedef Shape<(OperandShape::kW >> 2),  // one 3D tile per four elements in the K dimension
                (OperandShape::kH >> 4),  // four rows of SMEM per 64xK tile
                8,                        // Eight banks of MIO
                kAccessSize>
      VectorizedShape;

  /// Offset between acceses
  typedef Shape<1, 2 * WarpDelta::kW, 1, 1> Delta;

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
  typedef typename ShapeStrides<VectorizedShape, 1>::Shape Strides;

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Pointer count is always two
  static int const kPointerCount = 2;

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

  /// Shared memory load pointer
  Scalar const *pointer[kPointerCount];

  /// SMEM strides
  Coord<4> stride;

  /// Index in D dimension - needed to permute loads
  int k_index;

  //
  // Methods
  //

  /// Constructs a load iterator
  __device__ Volta884WarpMultiplicandLoadIterator(
      Params const &_params,
      Coord<4> const &_block_offset = make_Coord(0, 0, 0, 0),
      ThreadOffset offset_func = ThreadOffset())
      : stride(_params.stride), k_index(0) {
    CUTLASS_PRAGMA_UNROLL
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      Coord<4> offset = offset_func(ptr_idx);

      pointer[ptr_idx] = _params.pointer + (_block_offset + offset).template dot<int>(stride);
    }
  }

  /// Stores a fragment
  __device__ void load(Fragment &fragment, Coord<4> const &offset = make_Coord(0, 0, 0, 0)) const {
    FragmentIterator frag_iterator(fragment);

    // Iterate over each load
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          Coord<4> lds_offset =
              make_Coord(d, h, w, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kC);

          int ptr_idx = ((offset[0] >> 2) & 1);
          Scalar const *_pointer = pointer[ptr_idx];

          Load<typename Fragment::Element, VectorizedShape::kC, kMemorySpace>::load(
              reinterpret_cast<AccessType &>(frag_iterator.at(d, h, w, 0)),
              _pointer,
              stride.template dot<int>(lds_offset + offset));

          if (offset[0] & 2) {
            // peculiar swap for crosswise loads
            int lds128_idx = w + Iterations::kW * (h + Iterations::kH * d);
            uint64_t *left = reinterpret_cast<uint64_t *>(&fragment) + lds128_idx * 2;
            uint64_t *right = reinterpret_cast<uint64_t *>(&fragment) + lds128_idx * 2 + 1;
            uint64_t tmp = *left;
            *left = *right;
            *right = tmp;
          }
        }
      }
    }
  }

  /// Loads a fragment and increments to next K-index
  __device__ void load_post_increment(Fragment &fragment,
                                      Coord<4> const &offset = make_Coord(0, 0, 0, 0)) {
    load(fragment, offset + make_Coord(k_index, 0, 0, 0));
    ++k_index;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Store iterator specialized for B.column_major
template <
    /// Specifies threadblock tile shape
    typename Tile_,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    int WarpDelta>
struct Volta884ThreadblockMultiplicandStoreIterator<GemmOperand::kB,
                                                    MatrixLayout::kColumnMajor,
                                                    Tile_,
                                                    WarpCount,
                                                    WarpDelta> {
  //
  // Assertions
  //

  // Crosswise loaders may only span 32 x 128b along the K dimension
  static_assert(!(Tile_::kW % 8) && (Tile_::kW <= 256),
                "Tile dimensions must be divisible by 8 elements, and the K dimension may not span "
                "more than what a single warp can load");

  //
  // Constant and type definitions
  //

  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kB;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// Shape of thread-block multiplicand
  typedef Tile_ Tile;

  /// Number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp accumulator tiles along the outer dimension
  static int const kWarpDelta = WarpDelta;

  /// LDG.128 loads
  static int const kLdgAccessSize = 8;

  /// This implementation is specialized for 64b loads
  static int const kAccessSize = 4;

  /// Projects the threadblock tile
  typedef typename GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Stored tile has a structure designed for efficient MIO storing and loading
  typedef Shape<(OperandShape::kW >> 2),  // one 3D tile per four elements in the K dimension
                (OperandShape::kH >> 4),  // four rows of SMEM per 64xK tile
                16,                       // Sixteen banks of MIO
                kAccessSize>
      VectorizedShape;

  /// Offset between stores
  typedef Shape<WarpCount, 1, 1, 1> Delta;

  /// Shape of tile
  typedef Shape<1, 8, 4> WarpStoreCoverage;

  /// Number of iterations
  typedef Shape<
      // # of LDG.128s along the strided (outer) dimension
      OperandShape::kH / (WarpStoreCoverage::kH * kWarpCount),
      // # of LDG.128s along the contiguous (K) dimension
      OperandShape::kW / (WarpStoreCoverage::kW * kLdgAccessSize),
      // # STSs per LDG
      (kLdgAccessSize / kAccessSize),
      1>
      Iterations;

  /// Swizzled store iterator
  struct ThreadOffset {
    __device__ Coord<4> operator()(int ptr_idx) const {
      int warp_id = (threadIdx.x >> 5);
      int lane_id = (threadIdx.x & 0x1f);

      // Assumes a contiguous/blocked warp loading strategy
      int load_tile_idx = warp_id * Iterations::kD;

      // if Iterations::kD < 4, then we need to permute pointers
      if (Iterations::kD == 2) {
        ptr_idx ^= (warp_id & 1);
      }

      // Compute swizzled destination address
      int lane_w = (lane_id % WarpStoreCoverage::kW);
      int store_k_idx = lane_w * 2;

      int dest_tile_idx = load_tile_idx / 4;

      int dest_row = ((load_tile_idx >> 1) & 1);
      int dest_bank = (lane_id & 0x0f) ^ ((lane_id >> 4) & 1) ^ (ptr_idx << 1);

      Coord<4> offset = make_Coord(store_k_idx, dest_tile_idx * 2 + dest_row, dest_bank, 0);

      return offset;
    }
  };

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

  /// Store iterators require two pointers
  static int const kPointerCount = 2;

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

  /// Pointer to element type
  Scalar *pointer[kPointerCount];

  /// Strides
  Coord<4> stride;

  //
  // Methods
  //

  /// Constructs a store iterator
  __device__ Volta884ThreadblockMultiplicandStoreIterator(
      Params const &_params,
      Coord<4> const &_block_offset = make_Coord(0, 0, 0, 0),
      ThreadOffset offset_func = ThreadOffset())
      : stride(_params.stride) {
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      Coord<4> offset = offset_func(ptr_idx);
      pointer[ptr_idx] = _params.pointer + (_block_offset + offset).template dot<int>(stride);
    }
  }

  /// Stores a fragment
  CUTLASS_DEVICE
  void store(Fragment const &fragment, Coord<4> const &offset = make_Coord(0, 0, 0, 0)) const {
    FragmentConstIterator frag_iterator(fragment);

    // Iterate over each store
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {  // strided LDG.128s

      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {  // contiguous LDG.128s

        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {  // 2x STS operations per LDG

          int load_tile_idx = d;
          int k_idx = w + h * 8;
          int smem_row = (d >> 1);

          // Two store pointers
          int ptr_idx = ((load_tile_idx & 1) ^ ((load_tile_idx >> 1) & 1));

          Coord<4> sts_offset = make_Coord(k_idx, smem_row, 0, 0);

          if (true || (d == 0 && (threadIdx.x / 32) == 1)) {
            Store<typename Fragment::Element, kAccessSize, kMemorySpace>::store(
                reinterpret_cast<AccessType const &>(frag_iterator.at(d, h, w, 0)),
                pointer[ptr_idx],
                stride.template dot<int>(sts_offset + offset));
          }
        }
      }
    }
  }

  /// Increments store iterator to next tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &increment(int count = 1) {
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      pointer[ptr_idx] +=
          make_Coord(VectorizedShape::kD * count, 0, 0, 0).template dot<int>(stride);
    }
    return *this;
  }

  /// Increments to next tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &operator++() { return increment(); }

  __device__ Volta884ThreadblockMultiplicandStoreIterator &operator+=(int count) {
    return increment(count);
  }

  /// Increments store iterator to previous tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &decrement(int count = 1) {
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      pointer[ptr_idx] -=
          make_Coord(VectorizedShape::kD * count, 0, 0, 0).template dot<int>(stride);
    }
    return *this;
  }

  /// Increments to subsequent tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &operator--() { return decrement(); }

  /// Decrements to previous tile
  __device__ Volta884ThreadblockMultiplicandStoreIterator &operator-=(int count) {
    return decrement(count);
  }

  /// Stores a fragment and increments in the K dimension
  __device__ Volta884ThreadblockMultiplicandStoreIterator &store_post_increment(
      Fragment const &fragment, Coord<4> const &offset = make_Coord(0, 0, 0, 0)) {
    store(fragment, offset);
    return increment();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator to load a fragment for each warp-level tile specialized for B.column_major
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
                                            MatrixLayout::kColumnMajor,
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

  /// This implementation is specialized for 128b loads
  static int const kAccessSize = 8;

  /// Swizzled store iterator
  struct ThreadOffset {
    /// Compute thread offset coordinate for each pointer
    __device__ Coord<4> operator()(int ptr_idx) const {
      int warp_id = (threadIdx.x >> 5) / WarpDelta::kW;
      int lane_id = (threadIdx.x & 0x1f);

      int lane_in_quad = (lane_id & 0x3);
      int quad_id = ((lane_id >> 2) & 0x7);

      int oct_col_id = (quad_id >> 1);

      int oct_col = (oct_col_id & 1);
      int oct_left_right = ((oct_col_id >> 1) & 1) ^ (oct_col_id & 1) ^ ptr_idx;

      Coord<4> offset =
          make_Coord(0, warp_id * 2 + oct_col, (lane_in_quad * 2) + oct_left_right, 0);

      return offset;
    }
  };

  /// Projects the threadblock tile
  typedef typename GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Loaded tile has a structure designed for efficient MIO storing and loading
  typedef Shape<(OperandShape::kW >> 2),  // one 3D tile per four elements in the K dimension
                (OperandShape::kH >> 4),  // four rows of SMEM per 64xK tile
                8,                        // Eight banks of MIO
                kAccessSize>
      VectorizedShape;

  /// Offset between acceses
  typedef Shape<1, 2 * WarpDelta::kH, 1, 1> Delta;

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

  /// Pointer count is always two
  static int const kPointerCount = 2;

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

  /// Shared memory load pointer
  Scalar const *pointer[kPointerCount];

  /// SMEM strides
  Coord<4> stride;

  /// Index in D dimension - needed to permute loads
  int k_index;

  //
  // Methods
  //

  __device__ int column(uint16_t item) const { return ((item >> 8) & 0xff); }

  __device__ int column(half const *ptr) const {
    return column(reinterpret_cast<int16_t const &>(*ptr));
  }

  /// Constructs a load iterator
  __device__ Volta884WarpMultiplicandLoadIterator(
      Params const &_params,
      Coord<4> const &_block_offset = make_Coord(0, 0, 0, 0),
      ThreadOffset offset_func = ThreadOffset())
      : stride(_params.stride), k_index(0) {
    CUTLASS_PRAGMA_UNROLL
    for (int ptr_idx = 0; ptr_idx < kPointerCount; ++ptr_idx) {
      Coord<4> offset = offset_func(ptr_idx);

      pointer[ptr_idx] = _params.pointer + (_block_offset + offset).template dot<int>(stride);
    }
  }

  /// Stores a fragment
  __device__ void load(Fragment &fragment, Coord<4> const &offset = make_Coord(0, 0, 0, 0)) const {
    FragmentIterator frag_iterator(fragment);

    // Iterate over each load
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          Coord<4> lds_offset =
              make_Coord(d, h, w, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kC);

          int ptr_idx = ((offset[0] >> 2) & 1);
          Scalar const *_pointer = pointer[ptr_idx];

          Load<typename Fragment::Element, VectorizedShape::kC, kMemorySpace>::load(
              reinterpret_cast<AccessType &>(frag_iterator.at(d, h, w, 0)),
              _pointer,
              stride.template dot<int>(lds_offset + offset));

          if (offset[0] & 2) {
            // peculiar swap for crosswise loads
            int lds128_idx = w + Iterations::kW * (h + Iterations::kH * d);
            uint64_t *left = reinterpret_cast<uint64_t *>(&fragment) + lds128_idx * 2;
            uint64_t *right = reinterpret_cast<uint64_t *>(&fragment) + lds128_idx * 2 + 1;
            uint64_t tmp = *left;
            *left = *right;
            *right = tmp;
          }
        }
      }
    }
  }

  /// Loads a fragment and increments to next K-index
  __device__ void load_post_increment(Fragment &fragment,
                                      Coord<4> const &offset = make_Coord(0, 0, 0, 0)) {
    load(fragment, offset + make_Coord(k_index, 0, 0, 0));
    ++k_index;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
