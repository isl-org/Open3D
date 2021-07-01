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

  DO NOT INCLUDE THIS FILE DIRECTLY.

  This file is intended to be included by <cutlass/gemm/volta884_shared_tile.h> and defines
  partial specializations for templates specified therein.
*/

#pragma once

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for FP32 accumulator layouts
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue shared memory store iterator specialized for Volta's mma.sync.FP32 layout
template <
    /// Shape of warp-level GEMM
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Data type of accumulator elements
    typename Scalar_>
struct Volta884EpilogueSharedStoreIterator<WarpGemmTile_, WarpDelta_, Scalar_, float> {
  /// Warp-scoped GEMM tile size
  typedef WarpGemmTile_ WarpGemmTile;

  /// Tiling of warp elements across threadblock
  typedef WarpDelta_ WarpDelta;

  /// Scalar data type
  typedef Scalar_ Scalar;

  /// Accumulator data type (and layout)
  typedef float Accumulator;

  /// Index type
  typedef int Index;

  /// Index type
  typedef int LongIndex;

  // Host-side params
  struct Params {};

  /// Access size
  static int const kAccessSize = 1;

  /// Skew elements to ensure conflict free stores
  static int const kSkew = 2;

  /// Shape of one interleaved mma.sync tile
  typedef Shape<4, 32, 32> MmaTileShape;

  /// Four element fragment
  typedef Shape<WarpGemmTile::kW / MmaTileShape::kW, 1, 4, 1> Iterations;

  /// Delta separated by two elements
  typedef Shape<MmaTileShape::kW * WarpDelta::kW, 1, 2, 1> Delta;

  //
  // Dependent types
  //

  /// Predicate vector
  typedef PredicateVector<ShapeCount<Iterations>::kCount> PredicateVector;

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Fragment definition
  typedef Fragment<Scalar, ShapeCount<Iterations>::kCount * kAccessSize> Fragment;

  /// Elements loaded by one instruction
  typedef typename Vectorize<Scalar, kAccessSize>::Type AccessType;

  /// The fragment iterator.
  typedef FragmentIterator<Fragment, Iterations, AccessType> FragmentIterator;

  /// The fragment const iterator.
  typedef FragmentConstIterator<Fragment, Iterations, AccessType> FragmentConstIterator;

  /// Tensor reference type
  typedef TensorRef<Scalar, 4> TensorRef;

  //
  // Data members
  //

  /// Base pointer to SMEM allocation
  Scalar *pointer;

  /// Stride in shared memory
  Coord<4> strides;

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  Volta884EpilogueSharedStoreIterator(Params const &_params, TensorRef const &ref)
      : pointer(ref.data()), strides(make_Coord(1, WarpDelta::kW * WarpGemmTile::kW + kSkew, 1, 1)) {

    int warp_id = (threadIdx.x / kWarpSize);
    int lane_id = (threadIdx.x % kWarpSize);

    Coord<4> warp_idx = make_Coord(0, warp_id / WarpDelta::kW, warp_id % WarpDelta::kW, 0);

    Coord<4> warp_base = warp_idx * make_Coord(0, 4, MmaTileShape::kW, 0);

    Coord<4> thread_idx = make_Coord(0,
                                     (((lane_id >> 1) & 4) | (lane_id & 2)) >> 1,
                                     (lane_id & 1) | ((lane_id >> 1) & 8) | ((lane_id << 2) & 16),
                                     0);

    int offset = strides.template dot<int>(warp_base + thread_idx);

    pointer += offset;
  }

  /// Store to the epilogue tile.
  CUTLASS_DEVICE
  void store(Fragment const &fragment) const {
    FragmentConstIterator frag_iterator(fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          Coord<4> coord =
              make_Coord(d, h, w, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kC);

          int _offset = coord.template dot<int>(strides);

          Store<typename Fragment::Element, kAccessSize, kMemorySpace>::store(
              reinterpret_cast<AccessType const &>(frag_iterator.at(d, h, w, 0)), pointer,
             _offset);
        }
      }
    }
  }

  /// Stores to the epilogue tile - this iterator does not advance, so increment is null.
  CUTLASS_DEVICE
  void store_post_increment(Fragment const &fragment) { store(fragment); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue shared memory load iterator specialized for Volta's mma.sync.FP32 layout
template <
    /// Shape of warp-level GEMM
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Data type of accumulator elements
    typename Scalar_,
    /// Number of elements loaded per access
    int AccessSize_>
struct Volta884EpilogueSharedLoadIterator<WarpGemmTile_, WarpDelta_, Scalar_, AccessSize_, float> {
  /// Warp-scoped GEMM tile size
  typedef WarpGemmTile_ WarpGemmTile;

  /// Tiling of warp elements across threadblock
  typedef WarpDelta_ WarpDelta;

  /// Scalar data type
  typedef Scalar_ Scalar;

  /// Accumulator data type (and layout)
  typedef float Accumulator;

  /// Index type
  typedef int Index;

  /// Index type
  typedef int LongIndex;

  /// Number of elements accessed at once
  static int const kAccessSize = AccessSize_;

  /// Shape of one interleaved mma.sync tile
  typedef Shape<4, 32, 32> MmaTileShape;

  /// Total participating warps
  static int const kWarpCount = ShapeCount<WarpDelta>::kCount;

  /// Total participating threads
  static int const kThreadCount = kWarpCount * kWarpSize;

  /// Skew elements
  static int const kSkew = 2;

  /// This tile is to be strip-mined with a swizzling function
  typedef Shape<2 * WarpDelta::kH, 2, WarpGemmTile::kW * WarpDelta::kW, 1> Tile;

  /// Number of iterations
  typedef Shape<2 * WarpDelta::kH,
                (kThreadCount >= Tile::kW ? Tile::kH / (kThreadCount / Tile::kW) : Tile::kH),
                (kThreadCount >= Tile::kW ? 1 : Tile::kW / kThreadCount),
                1>
      Iterations;

  /// Delta between accesses
  typedef Shape<2, 1, kThreadCount, 1> Delta;

  //
  // Derived quantities
  //

  /// Predicate vector
  typedef PredicateVector<ShapeCount<Iterations>::kCount> PredicateVector;

  /// Fragment of elements to load
  typedef Fragment<Scalar, ShapeCount<Iterations>::kCount * kAccessSize> Fragment;

  /// Elements loaded by one instruction
  typedef typename Vectorize<Scalar, kAccessSize>::Type AccessType;

  /// The fragment iterator.
  typedef FragmentIterator<Fragment, Iterations, AccessType> FragmentIterator;

  /// The fragment const iterator.
  typedef FragmentConstIterator<Fragment, Iterations, AccessType> FragmentConstIterator;

  static_assert(!(kSkew % kAccessSize), "Access size must have compatible alignment with skew");

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Tensor reference type
  typedef TensorRef<Scalar, 4> TensorRef;

  /// Host-side params
  struct Params {};

  //
  // Data members
  //

  /// Pointer
  Scalar const *pointer;

  /// Strides
  Coord<4> strides;

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  Volta884EpilogueSharedLoadIterator(Params const &_params, TensorRef const &ref)
      : pointer(ref.data()),
        strides(make_Coord((WarpDelta::kW * WarpGemmTile::kW + kSkew) * kAccessSize,
                           (WarpDelta::kW * WarpGemmTile::kW + kSkew) * kAccessSize,
                           kAccessSize,
                           1)) {
    // strip-mine this tile
    int tid = threadIdx.x;

    int residual_w = (tid / (Tile::kW));
    int offset_w = (tid % (Tile::kW));

    int offset_h = (residual_w % Tile::kH);
    int offset_d = (residual_w / Tile::kH);

    Coord<4> offset = make_Coord(offset_d * Delta::kW, offset_h * Delta::kH, offset_w, 0);

    pointer += strides.template dot<int>(offset);
  }

  /// Loads a fragment from the epilogue tile.
  CUTLASS_DEVICE
  void load(Fragment &fragment) const {
    FragmentIterator frag_iterator(fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          Coord<4> coord =
              make_Coord(d, h, w, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kW);

          int _offset = coord.template dot<int>(strides);

          Load<typename Fragment::Element, kAccessSize, kMemorySpace>::load(
              reinterpret_cast<AccessType &>(frag_iterator.at(d, h, w, 0)), pointer, _offset);
        }
      }
    }
  }

  /// Loads a fragment - iterator does not actually advance, so increment operation is null.
  CUTLASS_DEVICE
  void load_post_increment(Fragment &fragment) { load(fragment); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for FP16 accumulator layouts
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue shared memory store iterator specialized for Volta's mma.sync.FP16 layout
template <
    /// Shape of warp-level GEMM
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Data type of accumulator elements
    typename Scalar_>
struct Volta884EpilogueSharedStoreIterator<WarpGemmTile_, WarpDelta_, Scalar_, half> {
  /// Warp-scoped GEMM tile size
  typedef WarpGemmTile_ WarpGemmTile;

  /// Tiling of warp elements across threadblock
  typedef WarpDelta_ WarpDelta;

  /// Scalar data type
  typedef Scalar_ Scalar;

  /// Accumulator data type (and layout)
  typedef half Accumulator;

  /// Index type
  typedef int Index;

  /// Index type
  typedef int LongIndex;

  /// Host-side params
  struct Params {};

  /// Dimensions of contiguous 32x32x4 Volta's mma.sync tile
  typedef Shape<4, 32, 32> MmaTileShape;

  /// Accumulator fragment
  typedef Shape<WarpGemmTile::kW / MmaTileShape::kW, 1, 2, 1> Iterations;

  /// Delta separated by two elements
  typedef Shape<MmaTileShape::kW * WarpDelta::kW, 1, 4, 1> Delta;

  /// Access size
  static int const kAccessSize = 1;

  /// Skew elements to ensure conflict free stores
  static int const kSkew = 2;

  /// Tensor reference type
  typedef TensorRef<Scalar, 4> TensorRef;

  //
  // Dependent types
  //

  /// Predicate vector
  typedef PredicateVector<ShapeCount<Iterations>::kCount> PredicateVector;

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Fragment definition
  typedef Fragment<Scalar, ShapeCount<Iterations>::kCount * kAccessSize> Fragment;

  /// Elements loaded by one instruction
  typedef typename Vectorize<Scalar, kAccessSize>::Type AccessType;

  /// The fragment iterator.
  typedef FragmentIterator<Fragment, Iterations, AccessType> FragmentIterator;

  /// The fragment const iterator.
  typedef FragmentConstIterator<Fragment, Iterations, AccessType> FragmentConstIterator;

  //
  // Data members
  //

  /// Base pointer to SMEM allocation
  Scalar *pointer;

  /// Stride in shared memory
  Coord<4> strides;

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  Volta884EpilogueSharedStoreIterator(Params const &_params, TensorRef const &ref)
      : pointer(ref.data()), strides(make_Coord(1, WarpGemmTile::kW * WarpDelta::kW + kSkew, 1, 1)) {

    int warp_id = (threadIdx.x / kWarpSize);
    int lane_id = (threadIdx.x % kWarpSize);

    int quad_id = (lane_id >> 2);
    int quadpair_id = (quad_id & 0x3);

    int quadpair_row = (quadpair_id & 1);
    int quadpair_col = (quadpair_id >> 1);
    int quad_hilo = (quad_id >> 2) & 1;

    int thread_row_offset = (quadpair_row * 2 + quad_hilo) * 8 + (lane_id & 3);
    int thread_col_offset = quadpair_col;

    Coord<4> thread_idx = make_Coord(0, thread_col_offset, thread_row_offset, 0);

    Coord<4> warp_base = make_Coord(0, warp_id / WarpDelta::kW, warp_id % WarpDelta::kW, 0) *
                         make_Coord(0, 2, kWarpSize, 0);
    Coord<4> offset = warp_base + thread_idx;

    pointer += strides.template dot<int>(offset);
  }

  /// Store to the epilogue tile.
  CUTLASS_DEVICE
  void store(Fragment const &fragment) const {
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          Coord<4> coord =
              make_Coord(d, h, w, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kC);

          int _offset = coord.template dot<int>(strides);

          Store<typename Fragment::Element, kAccessSize, kMemorySpace>::store(
              reinterpret_cast<AccessType const &>(fragment[w + Iterations::kW * d]),
              pointer,
              _offset);
        }
      }
    }
  }

  /// Stores to the epilogue tile - this iterator does not advance, so increment is null.
  CUTLASS_DEVICE
  void store_post_increment(Fragment const &fragment) { store(fragment); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue shared memory load iterator specialized for Volta's mma.sync.FP16 layout
template <
    /// Shape of warp-level GEMM
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Data type of accumulator elements
    typename Scalar_,
    /// Number of elements loaded per access
    int AccessSize_>
struct Volta884EpilogueSharedLoadIterator<WarpGemmTile_, WarpDelta_, Scalar_, AccessSize_, half> {
  /// Warp-scoped GEMM tile size
  typedef WarpGemmTile_ WarpGemmTile;

  /// Tiling of warp elements across threadblock
  typedef WarpDelta_ WarpDelta;

  /// Scalar data type
  typedef Scalar_ Scalar;

  /// Accumulator data type (and layout)
  typedef half Accumulator;

  /// Number of elements accessed at once
  static int const kAccessSize = AccessSize_;

  /// Shape of one interleaved mma.sync tile
  typedef Shape<4, 32, 32> MmaTileShape;

  /// This tile is to be strip-mined with a swizzling function
  typedef Shape<1, 2 * WarpDelta::kH, WarpGemmTile::kW * WarpDelta::kW / kAccessSize, kAccessSize>
      Tile;

  /// Index type
  typedef int Index;

  /// Index type
  typedef int LongIndex;

  /// Total participating warps
  static int const kWarpCount = ShapeCount<WarpDelta>::kCount;

  /// Number of participating threads
  static int const kThreadCount = kWarpSize * kWarpCount;

  /// Number of iterations
  typedef Shape<1,
                (kThreadCount >= Tile::kW ? Tile::kH / (kThreadCount / Tile::kW) : Tile::kH),
                (kThreadCount >= Tile::kW ? 1 : Tile::kW / kThreadCount),
                1>
      Iterations;

  /// Delta between thread-level accesses
  typedef typename platform::conditional<kThreadCount >= Tile::kW,
                                         Shape<1, (kThreadCount / Tile::kW), 1, 1>,
                                         Shape<1, 1, kThreadCount, 1> >::type Delta;

  //
  // Derived quantities
  //

  /// Predicate vector
  typedef PredicateVector<ShapeCount<Iterations>::kCount> PredicateVector;

  /// Fragment of elements to load
  typedef Fragment<Scalar, ShapeCount<Iterations>::kCount * kAccessSize> Fragment;

  /// Elements loaded by one instruction
  typedef typename Vectorize<Scalar, kAccessSize>::Type AccessType;

  /// The fragment iterator.
  typedef FragmentIterator<Fragment, Iterations, AccessType> FragmentIterator;

  /// The fragment const iterator.
  typedef FragmentConstIterator<Fragment, Iterations, AccessType> FragmentConstIterator;

  /// Skew elements
  static int const kSkew = 2;

  static_assert(!(kSkew % kAccessSize), "Access size must have compatible alignment with skew");

  /// Memory space access
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGeneric;

  /// Tensor reference type
  typedef TensorRef<Scalar, 4> TensorRef;

  /// Host-side params
  struct Params {};

  //
  // Data members
  //

  /// Pointer
  Scalar const *pointer;

  /// Strides
  Coord<4> strides;

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  Volta884EpilogueSharedLoadIterator(Params const &_params, TensorRef const &ref)
      : pointer(ref.data()),
        strides(make_Coord(2 * (WarpDelta::kW * WarpGemmTile::kW + kSkew) * kAccessSize,
                           (WarpDelta::kW * WarpGemmTile::kW + kSkew) * kAccessSize,
                           kAccessSize,
                           1)) {
    // strip-mine this tile
    Coord<4> offset = make_Coord(0, threadIdx.x / Tile::kW, threadIdx.x % Tile::kW, 0);

    pointer += strides.template dot<int>(offset);
  }

  /// Loads a fragment from the epilogue tile.
  CUTLASS_DEVICE
  void load(Fragment &fragment) const {
    FragmentIterator frag_iterator(fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          Coord<4> coord =
              make_Coord(d, h, w, 0) * make_Coord(Delta::kD, Delta::kH, Delta::kW, Delta::kW);

          int _offset = coord.template dot<int>(strides);

          Load<typename Fragment::Element, kAccessSize, kMemorySpace>::load(
              reinterpret_cast<AccessType &>(fragment[w + Iterations::kW * h]), pointer, _offset);
        }
      }
    }
  }

  /// Loads a fragment - iterator does not actually advance, so increment operation is null.
  CUTLASS_DEVICE
  void load_post_increment(Fragment &fragment) { load(fragment); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
