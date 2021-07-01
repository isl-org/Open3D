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
    \brief Defines the epilogue phase of the GEMM computation for IGEMM, supporting integer and
      floating-point output matrix formats.
*/
#pragma once

#include "cutlass/convert.h"
#include "cutlass/fragment.h"
#include "cutlass/gemm/gemm_global_stream.h"
#include "cutlass/gemm/gemm_shared_stream.h"
#include "cutlass/gemm/igemm_global_tile.h"
#include "cutlass/reshape_tile.h"
#include "cutlass/tile_iterator.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kElements_>
struct IgemmFloatToInt8Converter {
  /// The input fragment.
  typedef Fragment<float, kElements_> InputFragment;
  /// The output fragment.
  typedef Fragment<int8_t, kElements_> OutputFragment;

  // We are packing 4 floats into int32 registers so we need kElements to be multiple of 4.
  static_assert(kElements_ % 4 == 0, "kElements must be multiple of 4");

  /// Ctor.
  CUTLASS_DEVICE IgemmFloatToInt8Converter() {}

  /// Transform a fragment.
  CUTLASS_DEVICE void transform(InputFragment const& src, OutputFragment& dst) {
    transform(src, 0, dst);
  }

  /// Transform a fragment.
  template <typename Fragment_>
  CUTLASS_DEVICE void transform(Fragment_ const& src, int offset, OutputFragment& dst) {
    // The inputs.
    float4 const* src_f4 = reinterpret_cast<float4 const*>(&src[0]);
    // The outputs.
    int* dst_int = reinterpret_cast<int*>(&dst[0]);

    // Iterate over the floats and pack them together to produce ints.
    for (int i = 0; i < kElements_ / 4; ++i) {
      // Read the float4.
      float4 f4 = src_f4[i];

      // Clamp the 4 elements of the floats to the [-128, +127] range.
      float x = fmaxf(-128.f, fminf(127.f, f4.x));
      float y = fmaxf(-128.f, fminf(127.f, f4.y));
      float z = fmaxf(-128.f, fminf(127.f, f4.z));
      float w = fmaxf(-128.f, fminf(127.f, f4.w));

      // Convert to integers.
      int ix = (int)x;
      int iy = (int)y;
      int iz = (int)z;
      int iw = (int)w;

      // Extract the lower bytes to build an int32 with 4 int8.
      asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(ix) : "r"(iy));
      asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(iz) : "r"(iw));
      asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(ix) : "r"(iz));

      // Store the int.
      dst_int[i] = ix;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputScalar_, typename OutputFragment_>
struct IgemmGlobalStoreTransformer {
  typedef Convert<Fragment<InputScalar_, OutputFragment_::kElements>, OutputFragment_> Transformer;
};

template <int kElements_>
struct IgemmGlobalStoreTransformer<float, Fragment<int8_t, kElements_> > {
  typedef IgemmFloatToInt8Converter<kElements_> Transformer;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kElements_>
struct IgemmInt8ToFloatConverter {
  /// The input fragment.
  typedef Fragment<int8_t, kElements_> InputFragment;
  /// The output fragment.
  typedef Fragment<float, kElements_> OutputFragment;

  // We are unpacking 4 int8s from int32.
  static_assert(kElements_ % 4 == 0, "kElements must be multiple of 4");

  /// Ctor.
  CUTLASS_DEVICE IgemmInt8ToFloatConverter() {}

  /// Transform a fragment.
  CUTLASS_DEVICE void transform(InputFragment const& src, OutputFragment& dst) {
    transform(src, 0, dst);
  }

  /// Transform a fragment.
  template <typename Fragment_>
  CUTLASS_DEVICE void transform(Fragment_ const& src, int offset, OutputFragment& dst) {
    // The inputs.
    int const* src_int = reinterpret_cast<int const*>(&src[0]);
    // The outputs.
    float4* dst_f4 = reinterpret_cast<float4*>(&dst[0]);

    // Iterate over the int8 and unpack them together to produce floats.
    for (int i = 0; i < kElements_ / 4; ++i) {
      // Read the int.
      int ix, iy, iz, iw = src_int[i];

      // Extract the 4 bytes.
      asm volatile("prmt.b32 %0, 0x0, %1, 0x4440;" : "=r"(ix) : "r"(iw));
      asm volatile("prmt.b32 %0, 0x0, %1, 0x4441;" : "=r"(iy) : "r"(iw));
      asm volatile("prmt.b32 %0, 0x0, %1, 0x4442;" : "=r"(iz) : "r"(iw));
      asm volatile("prmt.b32 %0, 0x0, %1, 0x4443;" : "=r"(iw) : "r"(iw));

      // The floats.
      float fx, fy, fz, fw;

      // Convert to floats (make sure we generate I2F.F32.S8).
      asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fx) : "r"(ix));
      asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fy) : "r"(iy));
      asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fz) : "r"(iz));
      asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fw) : "r"(iw));

      // Store the float4.
      dst_f4[i] = make_float4(fx, fy, fz, fw);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputFragment_, typename OutputScalar_>
struct IgemmGlobalLoadTransformer {
  typedef Convert<InputFragment_, Fragment<OutputScalar_, InputFragment_::kElements> > Transformer;
};

template <int kElements_>
struct IgemmGlobalLoadTransformer<Fragment<int8_t, kElements_>, float> {
  typedef IgemmInt8ToFloatConverter<kElements_> Transformer;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputScalar_, typename OutputFragment_>
struct IgemmSharedStoreTransformer {
  typedef Convert<Fragment<InputScalar_, OutputFragment_::kElements>, OutputFragment_> Transformer;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename IgemmConfig_, typename EpilogueFunctor_, typename Index_>
struct IgemmEpilogueTraitsHelper
    : public GemmEpilogueTraitsHelper<IgemmConfig_, EpilogueFunctor_, Index_> {
  /// The base class.
  typedef GemmEpilogueTraitsHelper<IgemmConfig_, EpilogueFunctor_, Index_> Base;
  /// The config.
  typedef IgemmConfig_ IgemmConfig;

  /// The scalar type of the epilogue.
  typedef typename Base::Scalar Scalar;
  /// The iterations.
  typedef typename Base::Iterations Iterations;
  /// The iterations strides.
  typedef typename Base::Delta Delta;

  /// The traits class for the iterator.
  typedef typename Base::GlobalLoadTileTraits GlobalLoadTileTraits;
  /// The iterator to store to shared memory.
  typedef GemmGlobalIteratorCd<GlobalLoadTileTraits> GlobalLoadIteratorC;
  /// The fragment that needs to be produced by the load iterator.
  typedef typename GlobalLoadIteratorC::Fragment GlobalFragmentC;
  /// The transformer from loaded data to math fragment.
  typedef
      typename IgemmGlobalLoadTransformer<GlobalFragmentC, Scalar>::Transformer GlobalTransformerC;

  /// The traits class for the iterator.
  typedef typename Base::GlobalStoreTileTraits GlobalStoreTileTraits;
  /// The iterator to store to shared memory.
  typedef GemmGlobalIteratorCd<GlobalStoreTileTraits> GlobalStoreIteratorD;
  /// The fragment that needs to be passed to that store iterator.
  typedef typename GlobalStoreIteratorD::Fragment GlobalFragmentD;
  /// The transformer from accumulators to shared memory fragments.
  typedef
      typename IgemmGlobalStoreTransformer<Scalar, GlobalFragmentD>::Transformer GlobalTransformerD;

  /// The traits class for the shared iterator to store D to shared memory.
  typedef typename Base::SharedStoreTileTraits SharedStoreTileTraits;
  /// The shared iterator to store D to shared memory.
  typedef TileStoreIterator<SharedStoreTileTraits,
                            typename SharedStoreTileTraits::Scalar,
                            IteratorAdvance::kH,
                            MemorySpace::kGlobal>
      SharedStoreIteratorD;
  /// The fragment that needs to be passed to that store iterator.
  typedef typename SharedStoreIteratorD::Fragment SharedStoreFragmentD;
  /// The transformer from accumulators to shared memory fragments.
  typedef typename IgemmSharedStoreTransformer<typename IgemmConfig::Accumulators::Element,
                                               SharedStoreFragmentD>::Transformer
      SharedStoreTransformerD;
  /// The traits class for the shared iterator to load D from shared memory.
  typedef typename Base::SharedLoadTileTraits SharedLoadTileTraits;
  /// The shared iterator to load D from shared memory.
  typedef TileLoadIterator<SharedLoadTileTraits,
                           typename SharedLoadTileTraits::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared>
      SharedLoadIteratorD;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The config.
    typename IgemmConfig_,
    /// The functor to do the math in the epilogue.
    typename EpilogueFunctor_,
    /// The index.
    typename Index_ = int,
    /// The helper class to assemble the traits.
    typename Helper_ = IgemmEpilogueTraitsHelper<IgemmConfig_, EpilogueFunctor_, Index_> >
struct IgemmEpilogueTraits : public GemmEpilogueTraits<
                                 // The output tile.
                                 typename IgemmConfig_::OutputTile,
                                 // The accumulators.
                                 typename IgemmConfig_::Accumulators,
                                 // The global iterator for C.
                                 typename Helper_::GlobalLoadIteratorC,
                                 // The transformer for C.
                                 typename Helper_::GlobalTransformerC,
                                 // The transformer for D.
                                 typename Helper_::GlobalTransformerD,
                                 // The global iterator for D.
                                 typename Helper_::GlobalStoreIteratorD,
                                 // The iterator to store D to shared memory.
                                 typename Helper_::SharedStoreIteratorD,
                                 // The shared store transformer for D.
                                 typename Helper_::SharedStoreTransformerD,
                                 // The stream to load D from shared memory.
                                 typename Helper_::SharedLoadStreamD,
                                 // The iterations.
                                 typename Helper_::Iterations,
                                 // The strides between iterations.
                                 typename Helper_::Delta,
                                 // The functor to be used in the epilogue.
                                 EpilogueFunctor_,
                                 // The index.
                                 Index_> {
  /// Do we output in int8?
  static bool const kInt8Output =
      platform::is_same<typename IgemmConfig_::ScalarC, int8_t>::value != 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmEpilogueTraits_, bool = GemmEpilogueTraits_::kInt8Output>
struct IgemmEpilogue : public GemmEpilogue<GemmEpilogueTraits_> {
  /// The base class.
  typedef GemmEpilogue<GemmEpilogueTraits_> Base;

  /// Ctor.
  CUTLASS_DEVICE IgemmEpilogue(typename Base::Params const& params_,
                               typename Base::SharedStorage& shared_storage_,
                               Coord<3> const& _problem_size)
      : Base(params_, shared_storage_, _problem_size) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmEpilogueTraits_>
struct IgemmEpilogue<GemmEpilogueTraits_, true> : public GemmEpilogue<GemmEpilogueTraits_> {
  /// The base class.
  typedef GemmEpilogue<GemmEpilogueTraits_> Base;

  /// Ctor.
  CUTLASS_DEVICE IgemmEpilogue(typename Base::Params const& params_,
                               typename Base::SharedStorage& shared_storage_,
                               Coord<3> const& _problem_size)
      : Base(params_, shared_storage_, _problem_size) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
