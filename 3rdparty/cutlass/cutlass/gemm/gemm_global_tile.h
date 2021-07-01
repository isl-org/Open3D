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
    \brief Defines iterators for efficiently loading and storing to global memory.
*/
#pragma once

#include "cutlass/coord.h"
#include "cutlass/util/platform.h"

#include "cutlass/gemm/gemm_operand.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/reshape_tile.h"
#include "cutlass/tile_iterator.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

// The following functor reshapes a tile of threads to match a tile of data. The idea is that when
// the user wants to build the iterator traits, he/she may want to specify the tile independently
// from the number of scalars loaded/stored per instruction. For example, in the row-major version
// with a tile of size 128x8 - the user may want to that the iterator works with 32x8 threads if
// each thread loads 1 scalar per LDG. If the user changes to 4 scalars per LDG, then the tile of
// threads has to change. The code below detects that and correct the code automatically - it is
// a helper when the user does not specify the right configuration.

template <typename Tile_, typename Threads_, bool = (Tile_::kW < Threads_::kW)>
struct ReshapeThreads {
  typedef Threads_ Threads;
};

template <typename Tile_, typename Threads_>
struct ReshapeThreads<Tile_, Threads_, true> {
  typedef Shape<Threads_::kD, Threads_::kH * Threads_::kW / Tile_::kW, Tile_::kW, 1> Threads;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GemmOperand::Kind kOperand_,
          MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename Tile_,
          typename Threads_,
          int kAccessSize_>
struct GemmGlobalTileTraits {
  /// Identity of the operand
  static GemmOperand::Kind const kOperand = kOperand_;
  /// The layout.
  static MatrixLayout::Kind const kLayout = kLayout_;
  /// The scalar.
  typedef typename platform::remove_const<Scalar_>::type Scalar;
  /// The pointer.
  typedef Scalar_* Pointer;
  /// The number of scalars per LDG/STG.
  static int const kAccessSize = kAccessSize_;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kGlobal;
  /// The tile shape
  typedef Tile_ Tile;
  /// The vectorized tile shape
  typedef typename ReshapeTile<Tile_, kAccessSize_>::Tile VectorizedTile;
  /// The threads shape
  typedef typename ReshapeThreads<VectorizedTile, Threads_>::Threads Threads;
  /// The relative offset between two elements in the H/W dimension in adjacent threads.
  typedef Shape<1, 1, VectorizedTile::kC> ThreadsDelta;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, Threads::kH, Threads::kW * kAccessSize> Delta;

  /// Strides for immediate offset computation
  typedef Shape<0, 0, Threads::kW * ThreadsDelta::kW, kAccessSize> ImmediateOffsetStrides;
  /// The number of iterations needed to load/store the tile.
  typedef Shape<1,
                VectorizedTile::kH / Threads::kH,
                VectorizedTile::kW / Threads::kW,
                VectorizedTile::kC / kAccessSize>
      Iterations;

  typedef GemmMultiplicandTraits<Tile, kOperand, kLayout> MultiplicandTraits;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      int thread_offset_h = threadIdx.x / Threads::kW * ThreadsDelta::kH;
      int thread_offset_w = threadIdx.x % Threads::kW * ThreadsDelta::kW;

      return make_Coord(0, thread_offset_h, thread_offset_w, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, typename Tile_, typename Threads_, int kStrideH_, int kAccessSize_>
struct GemmGlobalTileCdTraits : public GemmGlobalTileTraits<GemmOperand::kC,
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

  /// The stride in the H dimension.
  static int const kStrideH = kStrideH_;
  /// Override the strides in each dimension between different loads/stores.
  typedef Shape<0, 0, Base::Delta::kW, Base::Delta::kC> Delta;

  typedef typename Base::Iterations Iterations;

  typedef typename Base::Threads Threads;

  typedef typename Base::ThreadsDelta ThreadsDelta;

  typedef typename Base::ImmediateOffsetStrides ImmediateOffsetStrides;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      int thread_offset_h = threadIdx.x / Threads::kW * kStrideH * Iterations::kH;
      int thread_offset_w = threadIdx.x % Threads::kW * ThreadsDelta::kW;

      return make_Coord(0, thread_offset_h, thread_offset_w, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TileTraits_, typename Index_ = int>
struct GemmGlobalIteratorAb
    : public TileLoadIterator<TileTraits_,
                              typename TileTraits_::Scalar,
                              TileTraits_::MultiplicandTraits::kKstrided ? IteratorAdvance::kH
                                                                         : IteratorAdvance::kW,
                              MemorySpace::kGlobal,
                              Index_> {
  /// This class.
  typedef GemmGlobalIteratorAb<TileTraits_, Index_> This_;  /// The base class.
  typedef TileLoadIterator<TileTraits_,
                           typename TileTraits_::Scalar,
                           TileTraits_::MultiplicandTraits::kKstrided ? IteratorAdvance::kH
                                                                      : IteratorAdvance::kW,
                           MemorySpace::kGlobal,
                           Index_>
      Base;
  /// The layout.
  static MatrixLayout::Kind const kLayout = TileTraits_::kLayout;
  /// The tile
  typedef typename TileTraits_::Tile Tile;
  /// Fragment type loaded by the iterator
  typedef typename Base::Fragment Fragment;
  /// The scalar.
  typedef typename TileTraits_::Scalar Scalar;
  /// The threads.
  typedef typename TileTraits_::Threads Threads;
  /// The index.
  typedef Index_ Index;
    /// Long index
  typedef long long LongIndex;
  /// The thread offset
  typedef typename TileTraits_::ThreadOffset ThreadOffset;
  /// Specifies in which dimension post-increment accesses advance.
  static IteratorAdvance::Kind const kAdvance = Base::kAdvance;

  typedef cutlass::PredicateVector<ShapeCount<typename Base::Iterations>::kCount> PredicateVector;

  /// Iterator parameters type
  typedef typename Base::Params BaseParams;

  struct Params : public BaseParams {
    /// Initializes params to load a strip-mined tile, given pointer and stride_h.
    CUTLASS_HOST_DEVICE int initialize(Scalar const* ptr,
                                       Index stride_d,
                                       Index stride_h) {
      return BaseParams::initialize(ptr, stride_d, stride_h, kAdvance == IteratorAdvance::kH ? 0 : 1);
    }
  };

  /// Offset of an individual lane from the start of the tile
  Coord<4> thread_offset;
  /// The parameters
  Params params;
  /// The predicates.
  PredicateVector predicates;

  CUTLASS_HOST_DEVICE void initialize_predicates(const Coord<3>& bounds, const Coord<3>& block_offset) {
    // Setup the masks to control loads.
    predicates.fill(0);

    // Fill in the bits of the predicate vector.
    for (int d = 0; d < Base::Iterations::kD; ++d) {
      for (int h = 0; h < Base::Iterations::kH; ++h) {
        for (int w = 0; w < Base::Iterations::kW; ++w) {
          for (int c = 0; c < Base::Iterations::kC; ++c) {
            bool flag = w * Base::Delta::kW + thread_offset[2] + block_offset[2] < bounds[2];
            if (kAdvance == IteratorAdvance::kH) {
              flag =
                  flag &&
                  (h * Base::Delta::kH + d * Base::Delta::kD) + thread_offset[1] + block_offset[1] <
                      bounds[1];
            } else {
              flag = flag && (h * Base::Delta::kH) + thread_offset[1] + block_offset[1] < bounds[1];
            }
            int const bit = ComputeOffsetFromShape<typename Base::Iterations>::get(d, h, w, c);
            predicates.set(bit, flag);
          }
        }
      }
    }
  }

  /// Ctor.
  CUTLASS_HOST_DEVICE GemmGlobalIteratorAb(Params const& _params,
                                           const Coord<3>& threadblock_offset,
                                           ThreadOffset thread_offset_func = ThreadOffset())
      : params(_params) {
    thread_offset = thread_offset_func();
    // Setup the pointer.
    params.pointer += ((threadblock_offset[1] + thread_offset[1]) * params.stride_h +
                       (threadblock_offset[2] + thread_offset[2]));

  }

  /// Increment the pointer in the W dimension.
  CUTLASS_HOST_DEVICE void inc_w() { Base::inc_w(); }
  /// Increment the pointer in the H dimension.
  CUTLASS_HOST_DEVICE void inc_h() { params.pointer += params.inc_h; }
  /// Increment the pointer in the D dimension.
  CUTLASS_HOST_DEVICE void inc_d() { params.pointer += params.inc_d; }
  /// Increment the pointer to move to the next iteration.
  CUTLASS_HOST_DEVICE void inc_advance() { params.pointer += params.inc_advance; }

  /// Loads a single fragment element from memory
  CUTLASS_HOST_DEVICE void load_element(
      typename Base::AccessType& value, int d, int h, int w, int c) const {
    int const offset =
        ComputeOffsetFromStrides<typename Base::ImmediateOffsetStrides>::get(0, 0, w, c);
    Load<Scalar,
         Base::kAccessSize,
         Base::kMemorySpace,
         Base::kFragmentElementType,
         typename Base::FragmentElement,
         Base::Tile::kW,
         Base::kAccessSize * sizeof(Scalar)>::load(value, params.pointer, offset);
  }

  /// That's the residue! Update the predicates.
  CUTLASS_HOST_DEVICE void residue(Index k) {
    // Update the predicate vector.
    for (int d = 0; d < Base::Iterations::kD; ++d) {
      for (int h = 0; h < Base::Iterations::kH; ++h) {
        for (int w = 0; w < Base::Iterations::kW; ++w) {
          for (int c = 0; c < Base::Iterations::kC; ++c) {
            Index offset = 0;
            if (kAdvance == IteratorAdvance::kH) {
              offset += thread_offset[1] + h * Base::Delta::kH + d * Base::Delta::kD;
            } else {
              offset += thread_offset[2] + w * Base::Delta::kW;
            }

            int const bit = ComputeOffsetFromShape<typename Base::Iterations>::get(d, h, w, c);
            if (offset >= k) {
              predicates.set(bit, false);
            }
          }
        }
      }
    }
  }

  /// Is the valid?
  CUTLASS_HOST_DEVICE bool valid(int d, int h, int w, int c) const {
    int const bit = ComputeOffsetFromShape<typename Base::Iterations>::get(d, h, w, c);
    return predicates[bit];
  }

  /// Adds a vector offset to the iterator
  CUTLASS_HOST_DEVICE GemmGlobalIteratorAb & operator+=(Coord<3> const &offset) {

    LongIndex _offset = offset.template dot<LongIndex>(
      make_Coord(params.stride_d, params.stride_h, params.stride_w)
    );

    params.pointer += _offset;
    return *this;
  }

  CUTLASS_HOST_DEVICE void add_pointer_offset(Index offset) { params.pointer += offset; }

  CUTLASS_HOST_DEVICE Index stride_advance(void) {
    Index stride = params.stride_h;
    if (kAdvance == IteratorAdvance::kW) {
      stride = params.stride_w;
    }
    return stride;
  }

  template <typename Fragment>
  CUTLASS_HOST_DEVICE void load_post_increment(Fragment& fragment) {
    typename Base::FragmentIterator frag_iterator(fragment);
    for (int d = 0; d < Base::Iterations::kD; ++d) {
      for (int h = 0; h < Base::Iterations::kH; ++h) {
        for (int w = 0; w < Base::Iterations::kW; ++w) {
          for (int c = 0; c < Base::Iterations::kC; ++c) {
            if (valid(d, h, w, c)) {
              load_element(
                  reinterpret_cast<typename Base::AccessType&>(frag_iterator.at(d, h, w, c)),
                  d,
                  h,
                  w,
                  c);
            }
          }
          if (w < Base::Iterations::kW - 1) {
            inc_w();
          }
        }
        if (h < Base::Iterations::kH - 1) {
          inc_h();
        }
      }
      if (d < Base::Iterations::kD - 1) {
        inc_d();
      }
    }
    inc_advance();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TileTraits_, typename Index_ = int>
struct GemmGlobalIteratorCd : public TileIteratorBase<TileTraits_,
                                                      typename TileTraits_::Scalar,
                                                      IteratorAdvance::kH,
                                                      MemorySpace::kGlobal,
                                                      Index_> {
  /// This class.
  typedef GemmGlobalIteratorCd<TileTraits_, Index_> This_;
  /// The base class.
  typedef TileIteratorBase<TileTraits_,
                           typename TileTraits_::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kGlobal,
                           Index_>
      Base;

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
    /// The index.
  typedef long long LongIndex;
  /// The thread offset
  typedef typename TileTraits_::ThreadOffset ThreadOffset;

  /// The params.
  struct Params {
    /// The pointer.
    Pointer pointer;
    /// The stride in the D dimension
    long long stride_d;
    /// The stride in the H dimension to setup the thread in the block.
    Index stride_h;
    /// The strides to increment the pointer.
    Index inc_advance, inc_h;
    /// The strides to increment the predicate offset
    Index predicate_inc_advance, predicate_inc_h;
    /// The column offset to compute the predicate for the columns.
    Index predicate_offset;

    /// Setup the params.
    CUTLASS_HOST_DEVICE int initialize(Pointer pointer,
                                       int stride_d_,
                                       Index ldm,
                                       Index bound,
                                       Index epilogue_stride_w,
                                       Index epilogue_delta_w) {
      // The pointer.
      this->pointer = pointer;
      // Stride per batch
      stride_d = stride_d_;
      // Each column of the matrix.
      stride_h = TileTraits_::ThreadsDelta::kH * ldm;
      // Each thread output 1 column per iteration. The stride between columns is given by the
      // number of scalars that are loaded per LDS for B.
      inc_h = ldm * TileTraits_::kStrideH;
      inc_advance =
          (ldm - ldm * TileTraits_::kStrideH * (Base::Iterations::kH - 1)) + epilogue_stride_w;

      predicate_offset = bound;
      predicate_inc_h = TileTraits_::kStrideH;
      predicate_inc_advance =
          -((TileTraits_::kStrideH * (Base::Iterations::kH - 1) - 1) + epilogue_delta_w);

      return 0;
    }

    CUTLASS_HOST_DEVICE int initialize(Pointer pointer, long long _stride_d, Index _stride_h, 
            Index _inc_advance, Index _inc_h, Index _predicate_inc_advance, Index _predicate_inc_h,
            Index _predicate_offset) {
      this->pointer = pointer;
      stride_d = _stride_d;
      stride_h = _stride_h;
      inc_advance = _inc_advance;
      inc_h = _inc_h;
      predicate_inc_advance = _predicate_inc_advance;
      predicate_inc_h = _predicate_inc_h;
      predicate_offset = _predicate_offset;

      return 0;
    }
  };

  /// Parameters.
  Params params;
  /// Offset of an individual lane from the start of the tile
  Coord<4> thread_offset;
  /// The predicates for the row.
  cutlass::PredicateVector<Base::Iterations::kW> predicates;
  
  /// Ctor.
  CUTLASS_HOST_DEVICE GemmGlobalIteratorCd(Params const& _params,
                                           const Coord<3>& bounds,
                                           const Coord<3>& block,
                                           int offset = 0,
                                           int pred_offset = 0,
                                           ThreadOffset thread_offset_func = ThreadOffset())
      : params(_params) {
    thread_offset = thread_offset_func();
    // Each warp works on a different column of the tile.
    int const h = thread_offset[1] + block[1];
    // Each lane writes a different element.
    int const w = thread_offset[2] + block[2];
    // Setup the pointer.
    params.pointer += ((h * params.stride_h + w) + offset);

    // Prepare the vector of predicates.
    for (int i = 0; i < Base::Iterations::kW; ++i) {
      predicates.set(i, w + i * Base::Delta::kW < bounds[2]);
    }
    params.predicate_offset -= (h + pred_offset);
  }

  /// Increment the pointer in the C dimension.
  CUTLASS_HOST_DEVICE void inc_c() {}
  /// Increment the pointer in the W dimension.
  CUTLASS_HOST_DEVICE void inc_w() {}
  /// Increment the pointer in the H dimension.
  CUTLASS_HOST_DEVICE void inc_h() {
    params.pointer += params.inc_h;
    params.predicate_offset -= params.predicate_inc_h;
  }
  /// Increment the pointer in the D dimension.
  CUTLASS_HOST_DEVICE void inc_d() {}
  /// Increment the pointer to move to the next iteration.
  CUTLASS_HOST_DEVICE void inc_advance() {
    params.pointer += params.inc_advance;
    params.predicate_offset -= params.predicate_inc_advance;
  }

  /// Adds a vector offset to the iterator
  CUTLASS_HOST_DEVICE GemmGlobalIteratorCd & operator+=(Coord<3> const &offset) {
    LongIndex _offset = offset.template dot<LongIndex>(
      make_Coord(params.stride_d, params.stride_h, 1)
    );
    params.pointer += _offset;
    return *this;
  }

  /// Loads a single fragment element from memory.
  CUTLASS_HOST_DEVICE void load_element(
      typename Base::AccessType& value, int d, int h, int w, int c) const {
    int const offset =
        ComputeOffsetFromStrides<typename Base::ImmediateOffsetStrides>::get(d, h, w, c);
    Load<Scalar,
         Base::kAccessSize,
         Base::kMemorySpace,
         Base::kFragmentElementType,
         typename Base::FragmentElement,
         Base::Tile::kW,
         Base::kAccessSize * sizeof(Scalar)>::load(value, params.pointer, offset);
  }

  /// Stores a single fragment element into memory.
  CUTLASS_HOST_DEVICE void store_element(
      typename Base::AccessType const& value, int d, int h, int w, int c) {
    int const offset =
        ComputeOffsetFromStrides<typename Base::ImmediateOffsetStrides>::get(d, h, w, c);
    Store<Scalar,
          Base::kAccessSize,
          Base::kMemorySpace,
          Base::kFragmentElementType,
          typename Base::FragmentElement,
          Base::Tile::kW,
          Base::kAccessSize * sizeof(Scalar)>::store(value, params.pointer, offset);
  }

  /// Test the validity of the
  CUTLASS_HOST_DEVICE bool valid(int d, int h, int w, int c) const {
    return predicates.at(w) && params.predicate_offset > 0;
  }

  /// add pointer offset
  CUTLASS_HOST_DEVICE void add_pointer_offset(LongIndex offset) { params.pointer += offset; }

  /// Loads and increments iterator
  template <typename Fragment>
  CUTLASS_HOST_DEVICE void load_post_increment(Fragment& fragment) {
    typename Base::FragmentIterator frag_iterator(fragment);
    for (int d = 0; d < Base::Iterations::kD; ++d) {
      for (int h = 0; h < Base::Iterations::kH; ++h) {
        for (int w = 0; w < Base::Iterations::kW; ++w) {
          for (int c = 0; c < Base::Iterations::kC; ++c) {
            if (valid(d, h, w, c)) {
              load_element(
                  reinterpret_cast<typename Base::AccessType&>(frag_iterator.at(d, h, w, c)),
                  d,
                  h,
                  w,
                  c);
            }
          }
          if (w < Base::Iterations::kW - 1) {
            inc_w();
          }
        }
        if (h < Base::Iterations::kH - 1) {
          inc_h();
        }
      }
      if (d < Base::Iterations::kD - 1) {
        inc_d();
      }
    }
    inc_advance();
  }

  template <typename Fragment>
  CUTLASS_HOST_DEVICE void store_post_increment(Fragment& fragment) {
    typename Base::FragmentIterator frag_iterator(fragment);
    for (int d = 0; d < Base::Iterations::kD; ++d) {
      for (int h = 0; h < Base::Iterations::kH; ++h) {
        for (int w = 0; w < Base::Iterations::kW; ++w) {
          for (int c = 0; c < Base::Iterations::kC; ++c) {
            if (valid(d, h, w, c)) {
              store_element(
                  reinterpret_cast<typename Base::AccessType&>(frag_iterator.at(d, h, w, c)),
                  d,
                  h,
                  w,
                  c);
            }
          }
          if (w < Base::Iterations::kW - 1) {
            inc_w();
          }
        }
        if (h < Base::Iterations::kH - 1) {
          inc_h();
        }
      }
      if (d < Base::Iterations::kD - 1) {
        inc_d();
      }
    }
    inc_advance();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
