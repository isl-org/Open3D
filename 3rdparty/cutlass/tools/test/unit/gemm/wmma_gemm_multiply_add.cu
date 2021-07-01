/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/wmma_matrix.h"

#ifdef CUTLASS_USE_WMMA_API

#include "cutlass_unit_tests.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/util/half.h"

#include "cutlass/gemm/gemm_global_stream.h"
#include "cutlass/gemm/gemm_shared_stream.h"
#include "cutlass/gemm/wmma_gemm_multiply_add.h"
#include "cutlass/gemm/wmma_gemm_global_tile.h"
#include "cutlass/gemm/wmma_gemm_shared_tile.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ProblemDesc {
  int m, n, k;
  inline __device__ ProblemDesc(int m_, int n_, int k_) : m(m_), n(n_), k(k_) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename StoreIterator_, typename LoadIterator_>
union SharedStorage {
  // Storage to store the data.
  typename StoreIterator_::SharedStorage store;
  // Storage to load the data.
  typename LoadIterator_::SharedStorage load;
};

template <class> struct Debug {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Threads_, int kW_, bool = (Threads_::kW > kW_)>
struct ReshapeThreadsA {
  typedef cutlass::Shape<Threads_::kD, Threads_::kH, Threads_::kW> Threads;
};

template <typename Threads_, int kW_>
struct ReshapeThreadsA<Threads_, kW_, true> {
  typedef cutlass::Shape<Threads_::kD, Threads_::kH * Threads_::kW / kW_, kW_> Threads;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Threads_, int kH_, bool = (Threads_::kW > kH_)>
struct ReshapeThreadsB {
  typedef cutlass::Shape<Threads_::kD, Threads_::kH, Threads_::kW> Threads;
};

template <typename Threads_, int kH_>
struct ReshapeThreadsB<Threads_, kH_, true> {
  typedef cutlass::Shape<Threads_::kD, Threads_::kH * Threads_::kW / kH_, kH_> Threads;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#if 1
template <typename Traits_>
static __global__ void kernel_nt(half const *d_a, int lda, half const *d_b, int ldb, float *d_c,
                                 int ldc) {
#if 0
  // The default configuration of threads.
  typedef cutlass::Shape<1, Warps_::kCount, 32> Threads_;
  // The threads.
  typedef typename ReshapeThreadsA<Threads_, OutputTile_::kW>::Threads ThreadsA;
  // The threads.
  typedef typename ReshapeThreadsB<Threads_, OutputTile_::kH>::Threads ThreadsB;
  // The number of elements loaded per LDG.
  int const kScalarsPerLdg = 1;
  // The tile for A.
  typedef cutlass::Shape<1, OutputTile_::kD, OutputTile_::kW> TileA;
  // The tile for B.
  typedef cutlass::Shape<1, OutputTile_::kD, OutputTile_::kH> TileB;
  // The tile for C.
  typedef cutlass::Shape<1, Warps_::kH*WmmaShape_::kH, OutputTile_::kW> TileC;
#endif

  // The problem descriptor.
  ProblemDesc desc(Traits_::OutputTile::kW, Traits_::OutputTile::kH, Traits::OutputTile::kD);

  // The elements computed by a single warp.
  typedef typename cutlass::ShapeDiv<OutputTile_, Warps_>::Shape AccumulatorsPerWarp;

  // Global memory load for A.
  typedef cutlass::gemm::GemmGlobalIteratorAb<
    cutlass::gemm::GemmGlobalIteratorTraits<
      cutlass::GemmOperand::kA, cutlass::MatrixLayout::kColumnMajor, half const, TileA, ThreadsA, kScalarsPerLdg> 
    >
    GlobalLoadIteratorA;

  // Shared store iterator for A.
  typedef cutlass::gemm::GemmSharedStoreIteratorAb<
    cutlass::gemm::GemmSharedStoreIteratorAbTraits<
      half, TileA, ThreadsA, kScalarsPerLdg> 
    >
    SharedStoreIteratorA;

  // The global stream for A.
  typedef cutlass::gemm::GlobalLoadStream<
    GlobalLoadIteratorA,
    cutlass::Copy<typename GlobalLoadIteratorA::Fragment>,
    SharedStoreIteratorA>
  GlobalLoadStreamA;

  // Shared load iterator for A.
  typedef cutlass::gemm::WmmaGemmSharedLoadIteratorA<
    cutlass::gemm::WmmaGemmSharedLoadIteratorAbTraits<
      cutlass::GemmOperand::kA, 
      cutlass::MatrixLayout::kColumnMajor, half, OutputTile_, Warps_, WmmaShape_> >
        SharedLoadIteratorA;

  // Global memory load for B.
  typedef cutlass::gemm::GemmGlobalIteratorAb<
    cutlass::gemm::GemmGlobalIteratorTraits<
      cutlass::GemmOperand::kB, cutlass::MatrixLayout::kRowMajor, half const, TileB, ThreadsB, kScalarsPerLdg> >
      GlobalLoadIteratorB;

  // Shared store iterator for B.
  typedef cutlass::gemm::GemmSharedStoreIteratorAb<
    cutlass::gemm::GemmSharedStoreIteratorAbTraits<
      half, TileB, ThreadsB, kScalarsPerLdg> >
      SharedStoreIteratorB;

  // The global stream for B.
  typedef cutlass::gemm::GlobalLoadStream<GlobalLoadIteratorB,
                                          cutlass::Copy<typename GlobalLoadIteratorB::Fragment>,
                                          SharedStoreIteratorB>
      GlobalLoadStreamB;

  // Shared load iterator for B.
  typedef cutlass::gemm::WmmaGemmSharedLoadIteratorB<
    cutlass::gemm::WmmaGemmSharedLoadIteratorAbTraits<
      cutlass::GemmOperand::kB, 
      cutlass::MatrixLayout::kRowMajor, half, OutputTile_, Warps_, WmmaShape_> >
      SharedLoadIteratorB;

  // Share memory to exchange data for A.
  __shared__ SharedStorage<GlobalLoadStreamA, SharedLoadIteratorA> shared_storage_a;

  // Share memory to exchange data for B.
  __shared__ SharedStorage<GlobalLoadStreamB, SharedLoadIteratorB> shared_storage_b;

  // Iterator to load A.
  typename GlobalLoadStreamA::Params global_params_a;
  global_params_a.initialize(desc, d_a, lda);
  GlobalLoadStreamA global_load_a(global_params_a, shared_storage_a.store, desc.m, desc.n, desc.k,
                                  cutlass::make_Coord(0, 0, 0));

  // Iterator to load B.
  typename GlobalLoadStreamB::Params global_params_b;
  global_params_b.initialize(desc, d_b, ldb);
  GlobalLoadStreamB global_load_b(global_params_b, shared_storage_b.store, desc.m, desc.n, desc.k,
                                  cutlass::make_Coord(0, 0, 0));

  // Load A/B.
  global_load_a.copy();
  global_load_b.copy();

  // Copy to shared memory.
  global_load_a.commit();
  global_load_b.commit();

  // Make sure the data is in shared memory.
  __syncthreads();

  // Load iterator A.
  typename SharedLoadIteratorA::Params shared_params_a;
  shared_params_a.initialize(desc);
  SharedLoadIteratorA shared_load_a(shared_params_a, shared_storage_a.load);

  // Load iterator B.
  typename SharedLoadIteratorB::Params shared_params_b;
  shared_params_b.initialize(desc);
  SharedLoadIteratorB shared_load_b(shared_params_b, shared_storage_b.load);

  // Copy A from shared memory.
  typename SharedLoadIteratorA::Fragment fragment_a;
  cutlass::gemm::load_shared(shared_load_a, fragment_a);

  // Copy B from shared memory.
  typename SharedLoadIteratorB::Fragment fragment_b;
  cutlass::gemm::load_shared(shared_load_b, fragment_b);

  // The functor to do WMMA.
  typedef cutlass::gemm::WmmaGemmMultiplyAdd<
    cutlass::MatrixLayout::kColumnMajor, 
    cutlass::MatrixLayout::kRowMajor, 
    cutlass::MatrixLayout::kColumnMajor, 
    float, 
    AccumulatorsPerWarp, 
    WmmaShape_> WmmaGemmMultiplyAdd;

  // The output fragment.
  typename WmmaGemmMultiplyAdd::Accumulators fragment_c;
  fragment_c.clear();

  // Do the WMMA.
  WmmaGemmMultiplyAdd multiply_add;
  multiply_add.multiply_add(fragment_a, fragment_b, fragment_c, fragment_c);

  // Global memory stream to store D.
  typedef cutlass::gemm::WmmaGemmGlobalIteratorCd<
    cutlass::gemm::WmmaGemmGlobalIteratorCdTraits<
      float, TileC, ThreadsA, 1> 
    >
    GlobalStoreIteratorD;
  typedef cutlass::gemm::GlobalStoreStream<GlobalStoreIteratorD> GlobalStoreStreamD;

  // The shared memory to store D.
  __shared__ typename GlobalStoreStreamD::SharedStorage shared_storage_stream_d;

  // Iterator to store C.
  typename GlobalStoreStreamD::Params global_params_d;
  global_params_d.initialize(desc, d_c, ldc);
  GlobalStoreStreamD global_store_d(global_params_d, shared_storage_stream_d, desc.m, desc.n, desc.k,
                                  cutlass::make_Coord(0, 0, 0));

  // Shared store iterator/stream for C.
  typedef cutlass::gemm::WmmaGemmSharedStoreIteratorD<
    cutlass::gemm::WmmaGemmSharedStoreIteratorDTraits<
      cutlass::MatrixLayout::kColumnMajor, float, OutputTile_, Warps_, WmmaShape_> >
    SharedStoreIteratorD;
  typedef cutlass::gemm::SharedStoreStream<SharedStoreIteratorD> SharedStoreStreamD;

  // Shared load iterator/stream for D.
  typedef cutlass::gemm::WmmaGemmSharedLoadIteratorD<
    cutlass::gemm::WmmaGemmSharedLoadIteratorDTraits<
      float, typename SharedStoreIteratorD::Tile, ThreadsA, 1> >
    SharedLoadIteratorD;
  typedef cutlass::gemm::SharedLoadStream<SharedLoadIteratorD> SharedLoadStreamD;

  // The shared memory structure to swizzle D.
  union SharedStorageD {
    typename SharedStoreStreamD::SharedStorage store;
    typename SharedLoadStreamD::SharedStorage load;
  };

  // The shared memory for D.
  __shared__ SharedStorageD shared_storage_d;

  // Store iterator D.
  typename SharedStoreStreamD::Params shared_store_params_d;
  shared_store_params_d.initialize();

  // Store iterator D.
  typename SharedLoadStreamD::Params shared_load_params_d;
  shared_load_params_d.initialize();

  // The number of WMMA in the tile H/W dimension (N/M in GEMM).
  int const kWmmaPerH = OutputTile_::kH / Warps_::kH / WmmaShape_::kH;
  int const kWmmaPerW = OutputTile_::kW / Warps_::kW / WmmaShape_::kW;

  // Iterate over the data.
  for (int i = 0; i < kWmmaPerH; ++i) {
      // Make sure the shared memory can be written to.
      __syncthreads();

      // Create the iterator to store to SMEM.
      SharedStoreStreamD shared_store_d(shared_store_params_d, 
                                        shared_storage_d.store, 
                                        fragment_c, 
                                        i*kWmmaPerW);
      shared_store_d.copy();
      shared_store_d.commit();

      // Make sure the shared memory was written.
      __syncthreads();

      // Create the iterator to load from SMEM.
      SharedLoadStreamD shared_load_d(shared_load_params_d, shared_storage_d.load);
      shared_load_d.copy();
      shared_load_d.commit();

      // Copy the data.
      cutlass::Copy<typename SharedLoadStreamD::Fragment> copy;
      copy.transform(shared_load_d.fragment(), global_store_d.fragment());

      // Copy the data to global memory.
      global_store_d.copy();
      global_store_d.commit();
  }
}
#else
template <typename OutputTile_, typename Warps_, typename WmmaShape_>
static __global__ void kernel_nt(half const *d_a, int lda, half const *d_b, int ldb, float *d_c,
                                 int ldc) {
  // The default configuration of threads.
  typedef cutlass::Shape<1, Warps_::kCount, 32> Threads_;
  // The threads.
  typedef typename ReshapeThreadsA<Threads_, OutputTile_::kW>::Threads ThreadsA;
  // The threads.
  typedef typename ReshapeThreadsB<Threads_, OutputTile_::kH>::Threads ThreadsB;
  // The number of elements loaded per LDG.
  int const kScalarsPerLdg = 1;
  // The tile for A.
  typedef cutlass::Shape<1, OutputTile_::kD, OutputTile_::kW> TileA;
  // The tile for B.
  typedef cutlass::Shape<1, OutputTile_::kD, OutputTile_::kH> TileB;
  // The tile for C.
  typedef cutlass::Shape<1, Warps_::kH*WmmaShape_::kH, OutputTile_::kW> TileC;

  // The problem descriptor.
  ProblemDesc desc(OutputTile_::kW, OutputTile_::kH, OutputTile_::kD);

  // The elements computed by a single warp.
  typedef typename cutlass::ShapeDiv<OutputTile_, Warps_>::Shape AccumulatorsPerWarp;

  // Global memory load for A.
  typedef cutlass::gemm::GemmGlobalIteratorAb<
    cutlass::gemm::GemmGlobalIteratorTraits<
      cutlass::GemmOperand::kA, cutlass::MatrixLayout::kColumnMajor, half const, TileA, ThreadsA, kScalarsPerLdg> 
    >
    GlobalLoadIteratorA;

  // Shared store iterator for A.
  typedef cutlass::gemm::GemmSharedStoreIteratorAb<
    cutlass::gemm::GemmSharedStoreIteratorAbTraits<
      half, TileA, ThreadsA, kScalarsPerLdg> 
    >
    SharedStoreIteratorA;

  // The global stream for A.
  typedef cutlass::gemm::GlobalLoadStream<
    GlobalLoadIteratorA,
    cutlass::Copy<typename GlobalLoadIteratorA::Fragment>,
    SharedStoreIteratorA>
  GlobalLoadStreamA;

  // Shared load iterator for A.
  typedef cutlass::gemm::WmmaGemmSharedLoadIteratorA<
    cutlass::gemm::WmmaGemmSharedLoadIteratorAbTraits<
      cutlass::GemmOperand::kA, 
      cutlass::MatrixLayout::kColumnMajor, half, OutputTile_, Warps_, WmmaShape_> >
        SharedLoadIteratorA;

  // Global memory load for B.
  typedef cutlass::gemm::GemmGlobalIteratorAb<
    cutlass::gemm::GemmGlobalIteratorTraits<
      cutlass::GemmOperand::kB, cutlass::MatrixLayout::kRowMajor, half const, TileB, ThreadsB, kScalarsPerLdg> >
      GlobalLoadIteratorB;

  // Shared store iterator for B.
  typedef cutlass::gemm::GemmSharedStoreIteratorAb<
    cutlass::gemm::GemmSharedStoreIteratorAbTraits<
      half, TileB, ThreadsB, kScalarsPerLdg> >
      SharedStoreIteratorB;

  // The global stream for B.
  typedef cutlass::gemm::GlobalLoadStream<GlobalLoadIteratorB,
                                          cutlass::Copy<typename GlobalLoadIteratorB::Fragment>,
                                          SharedStoreIteratorB>
      GlobalLoadStreamB;

  // Shared load iterator for B.
  typedef cutlass::gemm::WmmaGemmSharedLoadIteratorB<
    cutlass::gemm::WmmaGemmSharedLoadIteratorAbTraits<
      cutlass::GemmOperand::kB, 
      cutlass::MatrixLayout::kRowMajor, half, OutputTile_, Warps_, WmmaShape_> >
      SharedLoadIteratorB;

  // Share memory to exchange data for A.
  __shared__ SharedStorage<GlobalLoadStreamA, SharedLoadIteratorA> shared_storage_a;

  // Share memory to exchange data for B.
  __shared__ SharedStorage<GlobalLoadStreamB, SharedLoadIteratorB> shared_storage_b;

  // Iterator to load A.
  typename GlobalLoadStreamA::Params global_params_a;
  global_params_a.initialize(desc, d_a, lda);
  GlobalLoadStreamA global_load_a(global_params_a, shared_storage_a.store, desc.m, desc.n, desc.k,
                                  cutlass::make_Coord(0, 0, 0));

  // Iterator to load B.
  typename GlobalLoadStreamB::Params global_params_b;
  global_params_b.initialize(desc, d_b, ldb);
  GlobalLoadStreamB global_load_b(global_params_b, shared_storage_b.store, desc.m, desc.n, desc.k,
                                  cutlass::make_Coord(0, 0, 0));

  // Load A/B.
  global_load_a.copy();
  global_load_b.copy();

  // Copy to shared memory.
  global_load_a.commit();
  global_load_b.commit();

  // Make sure the data is in shared memory.
  __syncthreads();

  // Load iterator A.
  typename SharedLoadIteratorA::Params shared_params_a;
  shared_params_a.initialize(desc);
  SharedLoadIteratorA shared_load_a(shared_params_a, shared_storage_a.load);

  // Load iterator B.
  typename SharedLoadIteratorB::Params shared_params_b;
  shared_params_b.initialize(desc);
  SharedLoadIteratorB shared_load_b(shared_params_b, shared_storage_b.load);

  // Copy A from shared memory.
  typename SharedLoadIteratorA::Fragment fragment_a;
  cutlass::gemm::load_shared(shared_load_a, fragment_a);

  // Copy B from shared memory.
  typename SharedLoadIteratorB::Fragment fragment_b;
  cutlass::gemm::load_shared(shared_load_b, fragment_b);

  // The functor to do WMMA.
  typedef cutlass::gemm::WmmaGemmMultiplyAdd<
    cutlass::MatrixLayout::kColumnMajor, 
    cutlass::MatrixLayout::kRowMajor, 
    cutlass::MatrixLayout::kColumnMajor, 
    float, 
    AccumulatorsPerWarp, 
    WmmaShape_> WmmaGemmMultiplyAdd;

  // The output fragment.
  typename WmmaGemmMultiplyAdd::Accumulators fragment_c;
  fragment_c.clear();

  // Do the WMMA.
  WmmaGemmMultiplyAdd multiply_add;
  multiply_add.multiply_add(fragment_a, fragment_b, fragment_c, fragment_c);

  // Global memory stream to store D.
  typedef cutlass::gemm::WmmaGemmGlobalIteratorCd<
    cutlass::gemm::WmmaGemmGlobalIteratorCdTraits<
      float, TileC, ThreadsA, 1> 
    >
    GlobalStoreIteratorD;
  typedef cutlass::gemm::GlobalStoreStream<GlobalStoreIteratorD> GlobalStoreStreamD;

  // The shared memory to store D.
  __shared__ typename GlobalStoreStreamD::SharedStorage shared_storage_stream_d;

  // Iterator to store C.
  typename GlobalStoreStreamD::Params global_params_d;
  global_params_d.initialize(desc, d_c, ldc);
  GlobalStoreStreamD global_store_d(global_params_d, shared_storage_stream_d, desc.m, desc.n, desc.k,
                                  cutlass::make_Coord(0, 0, 0));

  // Shared store iterator/stream for C.
  typedef cutlass::gemm::WmmaGemmSharedStoreIteratorD<
    cutlass::gemm::WmmaGemmSharedStoreIteratorDTraits<
      cutlass::MatrixLayout::kColumnMajor, float, OutputTile_, Warps_, WmmaShape_> >
    SharedStoreIteratorD;
  typedef cutlass::gemm::SharedStoreStream<SharedStoreIteratorD> SharedStoreStreamD;

  // Shared load iterator/stream for D.
  typedef cutlass::gemm::WmmaGemmSharedLoadIteratorD<
    cutlass::gemm::WmmaGemmSharedLoadIteratorDTraits<
      float, typename SharedStoreIteratorD::Tile, ThreadsA, 1> >
    SharedLoadIteratorD;
  typedef cutlass::gemm::SharedLoadStream<SharedLoadIteratorD> SharedLoadStreamD;

  // The shared memory structure to swizzle D.
  union SharedStorageD {
    typename SharedStoreStreamD::SharedStorage store;
    typename SharedLoadStreamD::SharedStorage load;
  };

  // The shared memory for D.
  __shared__ SharedStorageD shared_storage_d;

  // Store iterator D.
  typename SharedStoreStreamD::Params shared_store_params_d;
  shared_store_params_d.initialize();

  // Store iterator D.
  typename SharedLoadStreamD::Params shared_load_params_d;
  shared_load_params_d.initialize();

  // The number of WMMA in the tile H/W dimension (N/M in GEMM).
  int const kWmmaPerH = OutputTile_::kH / Warps_::kH / WmmaShape_::kH;
  int const kWmmaPerW = OutputTile_::kW / Warps_::kW / WmmaShape_::kW;

  // Iterate over the data.
  for (int i = 0; i < kWmmaPerH; ++i) {
      // Make sure the shared memory can be written to.
      __syncthreads();

      // Create the iterator to store to SMEM.
      SharedStoreStreamD shared_store_d(shared_store_params_d, 
                                        shared_storage_d.store, 
                                        fragment_c, 
                                        i*kWmmaPerW);
      shared_store_d.copy();
      shared_store_d.commit();

      // Make sure the shared memory was written.
      __syncthreads();

      // Create the iterator to load from SMEM.
      SharedLoadStreamD shared_load_d(shared_load_params_d, shared_storage_d.load);
      shared_load_d.copy();
      shared_load_d.commit();

      // Copy the data.
      cutlass::Copy<typename SharedLoadStreamD::Fragment> copy;
      copy.transform(shared_load_d.fragment(), global_store_d.fragment());

      // Copy the data to global memory.
      global_store_d.copy();
      global_store_d.commit();
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename OutputTile_, typename Warps_, typename WmmaShape_>
void run() {
  /// Testbed type.
  typedef test::GemmTestbed<cutlass::half_t, cutlass::half_t, float, float, float> GemmTestbed;

  // Create the testbed.
  GemmTestbed testbed(OutputTile_::kW,  // M
                      OutputTile_::kH,  // N
                      OutputTile_::kD,  // K
                      cutlass::convert(cutlass::MatrixLayout::kColumnMajor),
                      cutlass::convert(cutlass::MatrixLayout::kRowMajor), 1, 0,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                      cutlass::convert(cutlass::MatrixLayout::kColumnMajor));

  // Initialize.
  testbed.initialize();

  // Launch the kernel.
  kernel_nt<OutputTile_, Warps_, WmmaShape_><<<1, 32*Warps_::kCount>>>(
      testbed.ptr_A(), testbed.lda(), 
      testbed.ptr_B(), testbed.ldb(), 
      testbed.ptr_computed(), testbed.ldc());
  ASSERT_EQ(cudaSuccess, cudaGetLastError());

  // Make sure it worked as expected.
  ASSERT_TRUE(testbed.verify_with_host());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_16x16x16_16x16x16) {
  run<cutlass::Shape<16, 16, 16>, cutlass::Shape<1, 1, 1>, cutlass::Shape<16, 16, 16> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_16x32x16_16x16x16) {
  run<cutlass::Shape<16, 32, 16>, cutlass::Shape<1, 1, 1>, cutlass::Shape<16, 16, 16> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_32x16x16_16x16x16) {
  run<cutlass::Shape<16, 16, 32>, cutlass::Shape<1, 1, 1>, cutlass::Shape<16, 16, 16> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_64x16x16_16x16x16) {
  run<cutlass::Shape<16, 16, 64>, cutlass::Shape<1, 1, 1>, cutlass::Shape<16, 16, 16> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_64x64x16_16x16x16) {
  run<cutlass::Shape<16, 64, 64>, cutlass::Shape<1, 1, 1>, cutlass::Shape<16, 16, 16> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_128x128x16_16x16x16) {
  run<cutlass::Shape<16, 128, 128>, cutlass::Shape<1, 2, 2>, cutlass::Shape<16, 16, 16> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_32x8x16_32x8x16) {
  run<cutlass::Shape<16, 8, 32>, cutlass::Shape<1, 1, 1>, cutlass::Shape<16, 8, 32> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_128x128x16_32x8x16) {
  run<cutlass::Shape<16, 128, 128>, cutlass::Shape<1, 2, 2>, cutlass::Shape<16, 8, 32> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_8x32x16_8x32x16) {
  run<cutlass::Shape<16, 32, 8>, cutlass::Shape<1, 1, 1>, cutlass::Shape<16, 32, 8> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm, multiply_add_f32_128x128x16_8x32x16) {
  run<cutlass::Shape<16, 128, 128>, cutlass::Shape<1, 2, 2>, cutlass::Shape<16, 32, 8> >();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // defined CUTLASS_USE_WMMA_API
