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

#include "cutlass/zip_fragment.h"
#include "cutlass/zip_tile_iterator.h"
#include "cutlass/util/complex.h"
#include "cutlass/gemm/volta884_gemm_epilogue_traits.h"
#include "cutlass/gemm/split_complex_linear_scaling.h"
#include "cutlass/util/pair.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Enables treating the accumulators selection as one object
template <typename First_, typename Second_>
struct ZipSelectAccumulators {

  /// Underlying selection function
  typedef First_ First;
  typedef Second_ Second;

  /// Accumulators
  typedef ZipFragment<
    typename First::Accumulators,
    typename Second::Accumulators> Accumulators;

  /// Fragment
  typedef ZipFragment<
    typename First::Fragment,
    typename Second::Fragment> Fragment;

  //
  // Data members
  //

  /// Selects the accumulators for the first part
  First first;

  /// Selects the accumulators for the second
  Second second;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_DEVICE
  ZipSelectAccumulators() { }

  /// Basic constructor
  CUTLASS_DEVICE
  ZipSelectAccumulators(First const &_first, Second const &_second): first(_first), second(_second) { }

  /// Selects accumulators for a given iteration of the epilogue
  CUTLASS_DEVICE
  Fragment operator()(Accumulators const &accum, Coord<2> const &idx) const {
    return make_ZipFragment(first(accum.first, idx), second(accum.second, idx));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines epilogue traits for complex-valued mma.sync GEMM
template <
  typename GemmConfig_,
  typename EpilogueFunctor_ = SplitComplexLinearScaling<typename GemmConfig_::MultiplyAdd::ScalarC>,
  typename Index_ = int>
struct Volta884ComplexGemmEpilogueTraits {

  /// GEMM configuration
  typedef GemmConfig_ GemmConfig;

  /// Epilogue functor
  typedef EpilogueFunctor_ Functor;

  /// Global memory mapping function
  typedef MatrixLayout::ColumnMajor GlobalDataLayout;

  /// Index type
  typedef Index_ Index;

  /// Long index used for offsets
  typedef long long LongIndex;

  /// Defines epilogue traits for real-valued Volta884 GEMM epilogue
  typedef typename Volta884GemmEpilogueTraitsHelper<
    GemmConfig,
    Functor,
    typename GemmConfig::MultiplyAdd::RealMultiplyAdd,
    Index>::EpilogueTraits RealEpilogueTraits;

  /// The output tile.
  typedef typename RealEpilogueTraits::OutputTile OutputTile;

  /// The warp-level GEMM tile
  typedef typename RealEpilogueTraits::WarpGemmTile WarpGemmTile;

  /// Tiling of warp accumulator elements
  typedef typename RealEpilogueTraits::WarpGemmTile WarpDelta;

  /// Multiply-add operation
  typedef typename GemmConfig::MultiplyAdd MultiplyAdd;

  /// The accumulators fragment type.
  typedef typename MultiplyAdd::Accumulators Accumulators;

  /// Selects a subset of accumulators for a given epilogue iteration
  typedef ZipSelectAccumulators<
    typename RealEpilogueTraits::SelectAccumulators,
    typename RealEpilogueTraits::SelectAccumulators> SelectAccumulators;

  /// The iterator to load source matrix from global memory.
  typedef cutlass::PredicatedTileLoadStream<
      ZipTileIterator<
        typename RealEpilogueTraits::GlobalLoadStreamC::Iterator,
        typename RealEpilogueTraits::GlobalLoadStreamC::Iterator
      >,
      typename RealEpilogueTraits::GlobalLoadStreamC::PredicateFunctor,
      ZipConvert<
        typename RealEpilogueTraits::GlobalLoadStreamC::Transformer,
        typename RealEpilogueTraits::GlobalLoadStreamC::Transformer
      >
    > GlobalLoadStreamC;

  /// The iterator to store the final GEMM computation to global memory.
  typedef cutlass::PredicatedTileStoreStream<
      ZipTileIterator<
        typename RealEpilogueTraits::GlobalStoreStreamD::Iterator,
        typename RealEpilogueTraits::GlobalStoreStreamD::Iterator
      >,
      typename RealEpilogueTraits::GlobalStoreStreamD::PredicateFunctor,
      ZipConvert<
        typename RealEpilogueTraits::GlobalStoreStreamD::Transformer,
        typename RealEpilogueTraits::GlobalStoreStreamD::Transformer
      >
    > GlobalStoreStreamD;

  /// The stream to store matrix product to shared memory
  typedef cutlass::TileStoreStream<
    ZipTileIterator<
      typename RealEpilogueTraits::SharedStoreStreamD::Iterator,
      typename RealEpilogueTraits::SharedStoreStreamD::Iterator
    >,
    ZipConvert<
      typename RealEpilogueTraits::SharedStoreStreamD::Transformer,
      typename RealEpilogueTraits::SharedStoreStreamD::Transformer
    >
  > SharedStoreStreamD;

  /// The stream to load the matrix product from shared memory
  typedef cutlass::TileLoadStream<
    ZipTileIterator<
      typename RealEpilogueTraits::SharedLoadStreamD::Iterator,
      typename RealEpilogueTraits::SharedLoadStreamD::Iterator
    >,
    ZipConvert<
      typename RealEpilogueTraits::SharedLoadStreamD::Transformer,
      typename RealEpilogueTraits::SharedLoadStreamD::Transformer
    >
  > SharedLoadStreamD;

  /// The scalar type of the source accumulator matrix.
  typedef typename RealEpilogueTraits::ScalarC ScalarC;

  /// The scalar type of the destination accumulator matrix.
  typedef typename RealEpilogueTraits::ScalarD ScalarD;

  //
  // Dependent types
  //

  /// Cover an entire warp-level tile
  typedef typename RealEpilogueTraits::Iterations Iterations;

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

    /// Stride for C
    platform::Pair<LongIndex, LongIndex> batch_stride_C;

    /// Stride for D
    platform::Pair<LongIndex, LongIndex> batch_stride_D;

    //
    // Methods
    //

    /// Default constructor
    CUTLASS_HOST_DEVICE
    Params() {
      batch_stride_C.first = 0;
      batch_stride_C.second = 0;

      batch_stride_D.first = 0;
      batch_stride_D.second = 0;
    }

    /// Setup the params.
    CUTLASS_HOST_DEVICE int initialize(
      platform::complex<typename Functor::Scalar> alpha,
      platform::complex<typename Functor::Scalar> beta,
      ScalarC const* real_C,
      Index real_ldc,
      ScalarC const* imag_C,
      Index imag_ldc,
      ScalarD* real_D,
      Index real_ldd,
      ScalarD* imag_D,
      Index imag_ldd) {

      int result = functor.initialize(alpha, beta);
      if (result) {
        return result;
      }

      // Setup the params for the global memory iterator for C.
      result = load_stream_c.iterator.first.initialize(
        real_C, real_ldc, real_ldc, 1);

      if (result) {
        return result;
      }

      result = load_stream_c.iterator.second.initialize(
        imag_C, imag_ldc, imag_ldc, 1);

      if (result) {
        return result;
      }

      // Setup the params for the global memory iterator for D.
      result = store_stream_d.iterator.first.initialize(
        real_D, real_ldd, real_ldd, 1);

      if (result) {
        return result;
      }

      result = store_stream_d.iterator.second.initialize(
        imag_D, imag_ldd, imag_ldd, 1);

      if (result) {
        return result;
      }

      return result;
    }

    /// Setup the params.
    CUTLASS_HOST_DEVICE int initialize(
      platform::complex<typename Functor::Scalar> alpha,
      platform::complex<typename Functor::Scalar> beta,
      ScalarC const* real_C,
      Index real_ldc,
      LongIndex stride_C_real,
      ScalarC const* imag_C,
      Index imag_ldc,
      LongIndex stride_C_imag,
      ScalarD* real_D,
      Index real_ldd,
      LongIndex stride_D_real,
      ScalarD* imag_D,
      Index imag_ldd,
      LongIndex stride_D_imag) {

      batch_stride_C.first = stride_C_real;
      batch_stride_C.second = stride_C_imag;

      batch_stride_D.first = stride_D_real;
      batch_stride_D.second = stride_D_imag;

      return initialize(alpha, beta, real_C, real_ldc, imag_C, imag_ldc, real_D, real_ldd, imag_D, imag_ldd);
    }
  };

  /// Shared memory buffer used by epilogue
  typedef ZipTileAllocation<
    typename RealEpilogueTraits::SharedStorage,
    typename RealEpilogueTraits::SharedStorage> SharedStorage;

  /// Functor computing the offset from the threadblock origin per iteration of
  /// the epilogue.
  typedef typename RealEpilogueTraits::GlobalOffset GlobalOffset;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm

namespace platform {

/// Here's a helpful arithmetic operator
CUTLASS_HOST_DEVICE
Pair<long long, long long> operator*(int s, Pair<long long, long long> _pair) {
  return Pair<long long, long long>(s * _pair.first, s * _pair.second);
}

}

}  // namespace cutlass

// clang-format on
