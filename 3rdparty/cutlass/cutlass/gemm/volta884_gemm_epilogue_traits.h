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
    \brief Implements the epilogue phase of the GEMM kernel that efficiently updates global memory
           with the computed matrix product.
*/

#pragma once

// clang-format off

#include "cutlass/tile_stream.h"
#include "cutlass/tile_allocation.h"

#include "cutlass/gemm/mma_shared_stream.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Abstraction to select accumulators from an accumulator tile for each iteration fo the epilogue
template <typename WarpGemmShape, typename WarpDelta, typename Scalar>
struct Volta884SelectAccumulators;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Selects accumulators from Volta mma.sync.F32 layout
template <typename WarpGemmShape_, typename WarpDelta_>
struct Volta884SelectAccumulators<WarpGemmShape_, WarpDelta_, float> {
  /// Shape of the warp-level matrix multiply operation
  typedef WarpGemmShape_ WarpGemmShape;

  /// Describes tiling of warp elements
  typedef WarpDelta_ WarpDelta;

  /// Data type of scalar
  typedef float Scalar;

  //
  // Derived types and constants
  //

  /// (Actual) number of accumulators held by each individual thread
  static int const kAccumulatorsPerThread = (WarpGemmShape::kH * WarpGemmShape::kW) / kWarpSize;

  /// Accumulators fragment
  typedef Fragment<Scalar, kAccumulatorsPerThread> Accumulators;

  /// Number of warps
  static int const kWarpCount = ShapeCount<WarpDelta>::kCount;

  /// Interleaved mma.sync shape
  typedef Shape<4, 32, 32> MmaTileShape;

  /// Hard-coded for FP32 layouts
  typedef Shape<1, WarpGemmShape::kW / MmaTileShape::kW, 4> Elements;

  /// Number of elements
  static int const kElements = ShapeCount<Elements>::kCount;

  /// Slice of accumulators
  typedef Fragment<Scalar, kElements> Fragment;

  //
  // Methods
  //

  /// Selects accumulators for a given iteration of the epilogue
  CUTLASS_DEVICE
  Fragment operator()(Accumulators const &accum, Coord<2> const &idx) const {
    Fragment frag;

    static int const kAccumPerOp = 8;

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Elements::kH; ++j) {

      // selects the 32x32 tile
      Coord<2> tile_32x32 = make_Coord(idx[0] / 8, j);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Elements::kW; ++i) {
        Coord<2> mma_op = make_Coord(((idx[0] >> 1) & 1), i / 2);

        int element = ((i & 1) << 1) | (idx[0] & 1) | (idx[0] & 4);

        int mma_op_idx = mma_op[1] + mma_op[0] * 2 + 4 * (tile_32x32[1] + 2 * tile_32x32[0]);

        frag[i + j * Elements::kW] = accum[element + kAccumPerOp * mma_op_idx];
      }
    }

    return frag;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Selects accumulators from Volta mma.sync.F16 layout
template <typename WarpGemmShape_, typename WarpDelta_>
struct Volta884SelectAccumulators<WarpGemmShape_, WarpDelta_, half> {
  /// Shape of the warp-level matrix multiply operation
  typedef WarpGemmShape_ WarpGemmShape;

  /// Describes tiling of warp elements
  typedef WarpDelta_ WarpDelta;

  /// Data type of accumulator elements
  typedef half Scalar;

  //
  // Derived types and constants
  //

  /// (Actual) number of accumulators held by each individual thread
  static int const kAccumulatorsPerThread = (WarpGemmShape::kH * WarpGemmShape::kW) / kWarpSize;

  /// Accumulators fragment
  typedef Fragment<Scalar, kAccumulatorsPerThread> Accumulators;

  /// Number of warps
  static int const kWarpCount = ShapeCount<WarpDelta>::kCount;

  /// Interleaved mma.sync shape
  typedef Shape<4, 32, 32> MmaTileShape;

  /// Hard-coded for FP16 layouts
  typedef Shape<1, WarpGemmShape::kW / MmaTileShape::kW, 2> Elements;

  /// Number of elements
  static int const kElements = ShapeCount<Elements>::kCount;

  /// Slice of accumulators
  typedef Fragment<Scalar, kElements> Fragment;

  //
  // Methods
  //

  /// Selects accumulators for a given iteration of the epilogue
  CUTLASS_DEVICE
  Fragment operator()(Accumulators const &accum, Coord<2> const &idx) const {
    Fragment frag;

    static int const kAccumPerOp = 8;

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Elements::kH; ++j) {

      // selects the 32x32 tile
      Coord<2> tile_32x32 = make_Coord(idx[0] / 16, j);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Elements::kW; ++i) {

        Coord<2> mma_op = make_Coord(((idx[0] >> 2) & 1), i & 1);

        int element = (idx[0] & 3) | ((idx[0] >> 1) & 4);

        int mma_op_idx = mma_op[1] + mma_op[0] * 2 + 4 * (tile_32x32[1] + 2 * tile_32x32[0]);

        frag[i + j * Elements::kW] = accum[element + kAccumPerOp * mma_op_idx];
      }
    }

    return frag;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The warp-level GEMM tile
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Size of vector to load or store
    int AccessSize,
    /// The accumulators fragment type - implies accumulator layout
    typename Accumulators_>
struct Volta884EpilogueGlobalTileTraits;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Global tile traits specialized for Volta mma.sync.F32 layout
template <
    /// The warp-level GEMM tile
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Size of vector to load or store
    int AccessSize>
struct Volta884EpilogueGlobalTileTraits<WarpGemmTile_, WarpDelta_, AccessSize, float> {
  /// Shape of warp-scoped GEMM tile
  typedef WarpGemmTile_ WarpGemmTile;

  /// Structure of MMA
  typedef WarpDelta_ WarpDelta;

  /// Access size of input/output elements
  static int const kAccessSize = AccessSize;

  /// Scalar type of accumulators - used to imply accumulator layout, not the data
  typedef float Accumulators;

  /// Strides for immediate offset computation
  typedef Shape<0, 0, 0, 0> ImmediateOffsetStrides;

  //typedef Shape<2, 2, 1, 1> Iterations;

  /// Hard-coded pitch between Volta mma.sync Quad Pair tiles
  static int const kMmaQuadPairWidth = 16;

  /// Hard-coded pitch between warp tiles
  static int const kInterleavedTileWidth = 32;

  /// Number of actual threads
  static int const kThreadCount = (WarpDelta::kH * WarpDelta::kW) * kWarpSize;

  /// Shape of the tile
  typedef Shape<2 * WarpDelta::kH, 2, WarpGemmTile::kW * WarpDelta::kW, 1> Tile;

  /// Number of iterations
  typedef Shape<2 * WarpDelta::kH,
    (kThreadCount >= Tile::kW ? Tile::kH / (kThreadCount / Tile::kW) : Tile::kH),
    (kThreadCount >= Tile::kW ? 1 : Tile::kW / kThreadCount),
    1> Iterations;

  /// Delta between accesses
  typedef Shape<kMmaQuadPairWidth, 2, WarpDelta::kW * kWarpSize, 1> Delta;

  /// Number of warps in threadblock
  static int const kWarpCount = ShapeCount<WarpDelta>::kCount;

  /// Custom thread-offset function
  struct ThreadOffset {
    CUTLASS_DEVICE
    Coord<4> operator()() {

      int tid = threadIdx.x;

      int residual_w = (tid / (Tile::kW));
      int offset_w = (tid % (Tile::kW));

      int offset_h = (residual_w % Tile::kH);
      int offset_d = (residual_w / Tile::kH);

      Coord<4> offset = make_Coord(offset_d * Delta::kD, offset_h * Delta::kH, offset_w, 0);

      return offset;
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Global tile traits specialized for Volta mma.sync.F16 layout
template <
    /// The warp-level GEMM tile
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Size of vector to load or store
    int AccessSize>
struct Volta884EpilogueGlobalTileTraits<WarpGemmTile_, WarpDelta_, AccessSize, half> {
  /// Shape of warp-scoped GEMM tile
  typedef WarpGemmTile_ WarpGemmTile;

  /// Structure of MMA tiles
  typedef WarpDelta_ WarpDelta;

  /// Access size of input/output elements
  static int const kAccessSize = AccessSize;

  /// Scalar type of accumulators - used to imply accumulator layout, not the data
  typedef half Accumulators;

  /// Hard-coded pitch between Volta mma.sync Quad Pair tiles
  static int const kMmaQuadPairWidth = 16;

  /// Hard-coded pitch between warp tiles
  static int const kInterleavedTileWidth = 32;

  /// Number of participating threads
  static int const kThreadCount = kWarpSize * WarpDelta::kH * WarpDelta::kW;

  /// Shape of the tile
  typedef Shape<1, 2 * WarpDelta::kH, WarpGemmTile::kW * WarpDelta::kW, 1> Tile;

  /// Strides for immediate offset computation
  typedef Shape<0, 0, 0, 0> ImmediateOffsetStrides;

  /// Number of iterations
  typedef Shape<
    1,
    (kThreadCount >= Tile::kW ? Tile::kH / (kThreadCount / Tile::kW) : Tile::kH),
    (kThreadCount >= Tile::kW ? 1 : Tile::kW / kThreadCount),
    1> Iterations;


  /// Delta between thread-level accesses
  typedef typename platform::conditional<
    kThreadCount >= Tile::kW,
    Shape<1, kMmaQuadPairWidth * (kThreadCount / Tile::kW), 1, 1>,
    Shape<1, kMmaQuadPairWidth, kThreadCount, 1>
    >::type Delta;

  /// Number of warps in threadblock
  static int const kWarpCount = ShapeCount<WarpDelta>::kCount;

  /// Custom thread-offset function
  struct ThreadOffset {
    CUTLASS_DEVICE
    Coord<4> operator()() {

      int tid = threadIdx.x;

      int residual_w = (tid / (Tile::kW));
      int offset_w = (tid % (Tile::kW));

      int offset_h = (residual_w % Tile::kH);
      int offset_d = (residual_w / Tile::kH);

      Coord<4> offset = make_Coord(offset_d * Delta::kD, offset_h * kMmaQuadPairWidth, offset_w, 0);

      return offset;
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Global offset functor for Volta884 epilogues
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename WarpDelta, typename AccumulatorType>
struct Volta884EpilogueGlobalOffset;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Functor computing the offset from the threadblock origin per iteration of
/// the epilogue. Specialized for Volta mma.sync.F32
template <typename WarpDelta>
struct Volta884EpilogueGlobalOffset<WarpDelta, float> {

  /// mma.sync instructions are arranged as spatially overlapping 32x32 tiles
  typedef Shape<4, 32, 32> MmaTileShape;

  CUTLASS_DEVICE
  Coord<3> operator()(Coord<2> const &iteration) const {

    int h = iteration[0];

    // C++ needs a better way to express bit swizzling
    int h_offset = ((h & 1) | ((h & 2) << 1) | (((h & 4) >> 2) * 8) |
                    (((h & 8) >> 3) * WarpDelta::kH * MmaTileShape::kH));

    return make_Coord(0, h_offset, iteration[1] * MmaTileShape::kW * WarpDelta::kW);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Functor computing the offset from the threadblock origin per iteration of
/// the epilogue. Specialized for Volta mma.sync.F16
template <typename WarpDelta>
struct Volta884EpilogueGlobalOffset<WarpDelta, half> {

  /// mma.sync instructions are arranged as spatially overlapping 32x32 tiles
  typedef Shape<4, 32, 32> MmaTileShape;

  CUTLASS_DEVICE
  Coord<3> operator()(Coord<2> const &iteration) const {

    int h = iteration[0];

    // C++ needs a better way to express bit swizzling
    int h_offset = (h & 15) | (h & 16) * 2 * WarpDelta::kH;

    Coord<3> offset = make_Coord(0, h_offset, iteration[1] * MmaTileShape::kW * WarpDelta::kW);
    return offset;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Epilogue traits for Volta884 epilogue
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue traits for Volta884 GEMMs
template <
    /// The threadblock GEMM tile
    typename OutputTile_,
    /// The warp-level GEMM tile
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// The accumulators fragment type.
    typename Accumulators_,
    /// Selects a slice of accumulators
    typename SelectAccumulators_,
    /// The iterator to load source matrix from global memory.
    typename GlobalLoadStreamC_,
    /// The iterator to store the final GEMM computation to global memory.
    typename GlobalStoreStreamD_,
    /// The stream to store matrix product to shared memory
    typename SharedStoreStreamD_,
    /// The stream to load the matrix product from shared memory
    typename SharedLoadStreamD_,
    /// The functor computing an element-wise operation on the matrix product
    typename Functor_,
    /// Global memory mapping function
    typename GlobalDataLayout_ = MatrixLayout::ColumnMajor,
    /// The index.
    typename Index_ = int>
struct Volta884EpilogueTraits {
  /// The output tile.
  typedef OutputTile_ OutputTile;

  /// The warp-level GEMM tile
  typedef WarpGemmTile_ WarpGemmTile;

  /// Tiling of warp accumulator elements
  typedef WarpDelta_ WarpDelta;

  /// The accumulators fragment type.
  typedef Accumulators_ Accumulators;

  /// Selects a subset of accumulators for a given epilogue iteration
  typedef SelectAccumulators_ SelectAccumulators;

  /// The iterator to load source matrix from global memory.
  typedef GlobalLoadStreamC_ GlobalLoadStreamC;

  /// The iterator to store the final GEMM computation to global memory.
  typedef GlobalStoreStreamD_ GlobalStoreStreamD;

  /// The stream to store matrix product to shared memory
  typedef SharedStoreStreamD_ SharedStoreStreamD;

  /// The stream to load the matrix product from shared memory
  typedef SharedLoadStreamD_ SharedLoadStreamD;

  /// The functor computing an element-wise operation on the matrix product
  typedef Functor_ Functor;

  /// Global memory mapping function
  typedef GlobalDataLayout_ GlobalDataLayout;

  /// The index.
  typedef Index_ Index;

  /// The scalar type of the source accumulator matrix.
  typedef typename GlobalLoadStreamC::Iterator::Scalar ScalarC;

  /// The scalar type of the destination accumulator matrix.
  typedef typename GlobalStoreStreamD::Iterator::Scalar ScalarD;

  //
  // Dependent types
  //

  static bool const kFp32Arrangement = sizeof(typename SelectAccumulators::Scalar) == 4;

  /// Skew elements
  static int const kSkew = 2;

  /// Number of columns of accumulators stored/loaded depends on the accumulator arrangement
  static int const kColumnsPerWarp = (kFp32Arrangement ? 4 : 2);

  /// mma.sync instructions are arranged as spatially overlapping 32x32 tiles
  typedef Shape<4, 32, 32> MmaTileShape;

  /// Cover an entire warp-level tile
  typedef Shape<1,
                WarpGemmTile::kH / kColumnsPerWarp,   // iterates over 32x32 accumulator tiles along N dimension
                1,                                    // iterates over 32x32 accumulator tiles along M dimension
                1>
      Iterations;

  /// Skew is needed to reduce bank conflicts to SMEM - this shape depends on accumulator layout
  typedef Shape<1,
    WarpDelta::kH * kColumnsPerWarp,                  // multiple columns in the gemm N dimension
    WarpDelta::kW * WarpGemmTile::kW + kSkew,         // rows in the gemm M dimension
    1
  > EpilogueTileAllocation;

  /// Parameters structure initialized on the host
  struct Params {
    /// The params for the C iterator.
    typename GlobalLoadStreamC::Params load_stream_c;

    /// The params for the D global iterator.
    typename GlobalStoreStreamD::Params store_stream_d;

    /// Epilogue functor params
    typename Functor::Params functor;

    /// The params for the D shared store iterator.
    typename SharedStoreStreamD::Params shared_store_stream_d;

    /// The params for the D shared load stream.
    typename SharedLoadStreamD::Params shared_load_stream_d;

    ///
    long long int batch_stride_C;

    ///
    long long int batch_stride_D;

    //
    // Methods
    //

    /// Default constructor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Helper constructor taking pointer, stride for source and destination matrices and functor
    /// params
    CUTLASS_HOST_DEVICE
    Params(ScalarD *ptr_D,
           int ldd,
           ScalarC const *ptr_C,
           int ldc,
           typename Functor::Params _functor = Functor::Params())
        : load_stream_c(), store_stream_d(), functor(_functor) {}

    /// Setup the params.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {
      batch_stride_C = desc.batch_stride_C;
      batch_stride_D = desc.batch_stride_D;

      // The parameters for the functor.
      int error_code = functor.initialize(desc);
      if (error_code) {
        return error_code;
      }

      // Setup the params for the global memory iterator for C.
      error_code = load_stream_c.iterator.initialize(
        desc.C.data(), desc.C.leading_dim(), desc.C.leading_dim(), 1
      );

      if (error_code) {
        return error_code;
      }

      // Setup the params for the global memory iterator for D.
      return store_stream_d.iterator.initialize(
        desc.D.data(), desc.D.leading_dim(), desc.D.leading_dim(), 1
      );
    }
  };

  /// Shared memory buffer used by epilogue
  typedef TileAllocation<
    typename SharedStoreStreamD::Iterator::Scalar,
    EpilogueTileAllocation> SharedStorage;

  /// Functor computing the offset from the threadblock origin per iteration of
  /// the epilogue.
  typedef Volta884EpilogueGlobalOffset<WarpDelta, typename SelectAccumulators::Scalar> GlobalOffset;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Volta884 Epilogue helper
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TileTraits, typename AccumulatorType>
struct Volta884EpiloguePredicateFunctor;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Functor specialized for the predicate arrangement in the Volta884 epilogue
template <typename TileTraits>
struct Volta884EpiloguePredicateFunctor<TileTraits, float> {
  /// Dimensions of the bounding volume
  Coord<3> bounds;

  /// Constructs a predicate functor given the bounds of a tensor
  CUTLASS_HOST_DEVICE
  Volta884EpiloguePredicateFunctor(Coord<3> _bounds) : bounds(_bounds) {}

  /// Computes the predicate given the logical position of an access
  CUTLASS_HOST_DEVICE
  bool operator()(Coord<3> const &iteration, Coord<3> const &offset) const {
    return
      (iteration[0] * TileTraits::Delta::kD + iteration[1] * TileTraits::Delta::kH +
        offset[1] < bounds[1]) &&
      (iteration[2] * TileTraits::Delta::kW + offset[2] < bounds[2]);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Functor specialized for the predicate arrangement in the Volta884 epilogue
template <typename TileTraits>
struct Volta884EpiloguePredicateFunctor<TileTraits, half> {
  /// Dimensions of the bounding volume
  Coord<3> bounds;

  /// Constructs a predicate functor given the bounds of a tensor
  CUTLASS_HOST_DEVICE
  Volta884EpiloguePredicateFunctor(Coord<3> _bounds) : bounds(_bounds) {}

  /// Computes the predicate given the logical position of an access
  CUTLASS_HOST_DEVICE
  bool operator()(Coord<3> const &iteration, Coord<3> const &offset) const {
    return iteration[1] * TileTraits::Delta::kH + offset[1] < bounds[1] &&
      iteration[2] * TileTraits::Delta::kW + offset[2] < bounds[2];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Volta884 Epilogue helper
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to define the traits for a Volta884 Epilogue
template <
  typename GemmConfig_,
  typename EpilogueFunctor_,
  typename MultiplyAdd_ = typename GemmConfig_::MultiplyAdd,
  typename Index_ = int>
struct Volta884GemmEpilogueTraitsHelper {

  /// Configuration object defining GEMM properties
  typedef GemmConfig_ GemmConfig;

  /// Warp-level tile
  typedef typename GemmConfig::AccumulatorsPerWarp WarpGemmShape;

  /// Warp delta
  typedef typename ShapeDiv<
    typename GemmConfig::OutputTile,
    WarpGemmShape>::Shape WarpDelta;

  /// Thread-block scoped tile
  typedef typename cutlass::ShapeMul<
    WarpGemmShape,
    WarpDelta
  >::Shape OutputTile;

  /// Multiply-add operation
  typedef MultiplyAdd_ MultiplyAdd;

  /// Epilogue functor
  typedef EpilogueFunctor_ Functor;

  /// Traits for global tile access
  typedef cutlass::gemm::Volta884EpilogueGlobalTileTraits<
    WarpGemmShape,
    WarpDelta,
    1,
    typename MultiplyAdd::ScalarC
  > EpilogueGlobalTileTraits;

  /// Iterator to load a slice of the C matrix from global memory
  typedef cutlass::TileLoadIterator<
    EpilogueGlobalTileTraits,
    typename GemmConfig::ScalarC,
    cutlass::IteratorAdvance::kW,
    cutlass::MemorySpace::kGlobal
  > TileLoadIteratorC;

  /// Conversion from C data type to accumulator data type
  typedef Convert<
    typename TileLoadIteratorC::Fragment,
    Fragment<typename MultiplyAdd::ScalarC, TileLoadIteratorC::Fragment::kElements>
    > ConvertSourceFragment;

  /// Iterator to store a slice of the D matrix to global memory
  typedef cutlass::TileStoreIterator<
    EpilogueGlobalTileTraits,
    typename GemmConfig::ScalarD,
    cutlass::IteratorAdvance::kW,
    cutlass::MemorySpace::kGlobal
  > TileStoreIteratorD;

  /// Conversion from accumulator data type to D data type
  typedef Convert<
    Fragment<typename MultiplyAdd::ScalarC, TileStoreIteratorD::Fragment::kElements>,
    typename TileStoreIteratorD::Fragment
    > ConvertDestinationFragment;

  /// Defines traits for an epilogue of a Volta884 GEMM
  typedef cutlass::gemm::Volta884EpilogueTraits<
    OutputTile,
    WarpGemmShape,
    WarpDelta,
    typename MultiplyAdd::Accumulators,
    cutlass::gemm::Volta884SelectAccumulators<
      WarpGemmShape,
      WarpDelta,
      typename MultiplyAdd::ScalarC
    >,
    cutlass::PredicatedTileLoadStream<
      TileLoadIteratorC,
      cutlass::gemm::Volta884EpiloguePredicateFunctor<
        EpilogueGlobalTileTraits,
        typename MultiplyAdd::ScalarC>,
      ConvertSourceFragment
    >,
    cutlass::PredicatedTileStoreStream<
      TileStoreIteratorD,
      cutlass::gemm::Volta884EpiloguePredicateFunctor<
        EpilogueGlobalTileTraits,
        typename MultiplyAdd::ScalarC>,
      ConvertDestinationFragment
    >,
    cutlass::TileStoreStream<
      cutlass::gemm::Volta884EpilogueSharedStoreIterator<
        WarpGemmShape,
        WarpDelta,
        typename MultiplyAdd::ScalarC,
        typename MultiplyAdd::ScalarC
      >
    >,
    cutlass::TileLoadStream<
      cutlass::gemm::Volta884EpilogueSharedLoadIterator<
        WarpGemmShape,
        WarpDelta,
        typename MultiplyAdd::ScalarC,
        1,
        typename MultiplyAdd::ScalarC
      >
    >,
    Functor
  > EpilogueTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

// clang-format on
