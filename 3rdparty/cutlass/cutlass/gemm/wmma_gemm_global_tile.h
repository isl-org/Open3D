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
    \brief Defines tile iterator traits for loading thread block-level tile from global memory.
*/
#pragma once

#include "cutlass/gemm/gemm_global_tile.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, typename Tile_, typename Threads_, int kAccessSize_>
struct WmmaGemmGlobalIteratorCdTraits : public GemmGlobalTileTraits<GemmOperand::kC,
                                                                    MatrixLayout::kColumnMajor,
                                                                    Scalar_,
                                                                    Tile_,
                                                                    Threads_,
                                                                    kAccessSize_> {
  /// The base class.
  typedef GemmGlobalTileTraits<GemmOperand::kC,
                               MatrixLayout::kColumnMajor,
                               Scalar_,
                               Tile_,
                               Threads_,
                               kAccessSize_>
      Base;

  /// Override the strides in each dimension between different loads/stores.
  typedef Shape<0, 0, Base::Delta::kW, Base::Delta::kC> Delta;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      int thread_offset_h = threadIdx.x / Base::Threads::kW;
      int thread_offset_w = threadIdx.x % Base::Threads::kW * Base::ThreadsDelta::kW;

      return make_Coord(0, thread_offset_h, thread_offset_w, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TileTraits_, typename Index_ = int>
struct WmmaGemmGlobalIteratorCd : public GemmGlobalIteratorCd<TileTraits_, Index_> {
  /// This class.
  typedef WmmaGemmGlobalIteratorCd<TileTraits_, Index_> This_;
  /// The traits.
  typedef TileTraits_ Traits;
  /// The base class.
  typedef GemmGlobalIteratorCd<Traits, Index_> Base;
  /// Override the strides in each dimension between different loads/stores.
  typedef Shape<0, 0, Base::Delta::kW, Base::Delta::kC> ImmediateOffsetStrides;
  /// The layout.
  static MatrixLayout::Kind const kLayout = TileTraits_::kLayout;

  /// The scalar.
  typedef typename TileTraits_::Scalar Scalar;
  /// The pointer.
  typedef typename TileTraits_::Pointer Pointer;
  /// The threads.
  typedef typename TileTraits_::Threads Threads;
  /// The index.
  typedef Index_ Index;
  /// The thread offset functor.
  typedef typename TileTraits_::ThreadOffset ThreadOffset;
  /// Base parameters.
  typedef typename Base::Params BaseParams;

  /// The params.
  struct Params : public BaseParams {
    /// Setup the params.
    CUTLASS_HOST_DEVICE int initialize(Pointer pointer,
                                       long long batch_stride,
                                       Index ldm,
                                       Index n,
                                       Index epilogue_stride_w,
                                       Index epilogue_delta_w) {
      // The pointer.
      BaseParams::pointer = pointer;
      // Stride between GEMMs
      this->stride_d = batch_stride;
      // Setup the base stride. One "group of threads" per column.
      this->stride_h = ldm;
      // Each thread output 1 column per iteration. .
      this->inc_h = ldm * TileTraits_::Threads::kH;
      this->inc_advance = this->inc_h + epilogue_stride_w;

      this->predicate_offset = n;
      this->predicate_inc_h = TileTraits_::Threads::kH;
      this->predicate_inc_advance = this->predicate_inc_h + epilogue_delta_w;

      return 0;
    }
  };

  /// Ctor.
  CUTLASS_DEVICE WmmaGemmGlobalIteratorCd(Params const& params,
                                          const Coord<3>& bounds,
                                          const Coord<3>& block,
                                          int const pointer_offset = 0,
                                          int const pred_offset = 0,
                                          ThreadOffset thread_offset_func = ThreadOffset())

      : Base(params, bounds, block, pointer_offset, pred_offset, thread_offset_func) {}

  /// Loads a single fragment element from memory
  CUTLASS_DEVICE void load_element(
      typename Base::AccessType& value, int d, int h, int w, int c) const {
    Base::load_element(value, d, h, w, c);
  }

  /// Stores a single fragment element into memory
  CUTLASS_DEVICE void store_element(
      typename Base::AccessType const& value, int d, int h, int w, int c) {
    int const offset =
        ComputeOffsetFromStrides<typename Base::ImmediateOffsetStrides>::get(d, h, w, 0);
    Store<Scalar,
          Base::kAccessSize,
          Base::kMemorySpace,
          Base::kFragmentElementType,
          typename Base::FragmentElement,
          Base::Tile::kW>::store(value, Base::params.pointer, offset);
  }

 public:
  template <typename Fragment>
  CUTLASS_DEVICE void load_post_increment(Fragment& fragment) {
    Base::load_post_increment(fragment);
  }

  template <typename Fragment>
  CUTLASS_DEVICE void store_post_increment(Fragment& fragment) {
    Base::store_post_increment(fragment);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
