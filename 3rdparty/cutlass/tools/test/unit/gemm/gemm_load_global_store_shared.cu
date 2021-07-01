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
#include "cutlass_unit_tests.h"
#include "tools/util/host_tensor.h"
#include "tools/test/unit/core/layout_verification.h"
#include "tools/util/tensor_view_io.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/shape.h"
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/gemm/dgemm_traits.h"
#include "cutlass/gemm/hgemm_traits.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

    // M/N/K struct.
    struct GemmDesc { 
      int m, n, k; 
      CUTLASS_HOST_DEVICE GemmDesc(int m_, int n_, int k_) : m(m_), n(n_), k(k_) {} 
    };

 /// Simple test to load from global memory and store to shared memory

    // Loading from global memory and storing to shared memory for A
    template <typename Traits>
    __global__ void Gemm_load_global_store_shared_a(
        typename Traits::GlobalLoadStreamA::Scalar *output,
        typename Traits::GlobalLoadStreamA::Scalar const *input,
        int M,
        int N,
        int K,
        int ldm) {

        //Create shared memory.
        __shared__ typename Traits::SharedStorage shared_storage;

        // Create those iterators.
        typedef typename Traits::GlobalLoadStreamA GlobalLoadStreamA;

        typename GlobalLoadStreamA::Params global_load_params;
        GemmDesc desc(M, N, K);
        global_load_params.initialize(desc, input, ldm);

        GlobalLoadStreamA stream_a(global_load_params, shared_storage.main_loop.stream_a.global, M, N, K, cutlass::make_Coord(0, 0, 0));
        stream_a.copy();
        stream_a.commit();

        // store barrier
        __syncthreads();

        // one thread writes everything out
        if (threadIdx.x == 0) {
            for (int i = 0; i < M*K; ++i) {
                output[i] = shared_storage.main_loop.stream_a.shared[i];
            }
        }

    }

    // Loading from global memory and storing to shared memory for B
    template <typename Traits>
    __global__ void Gemm_load_global_store_shared_b(
        typename Traits::GlobalLoadStreamB::Scalar *output,
        typename Traits::GlobalLoadStreamB::Scalar const *input,
        int M,
        int N,
        int K,
        int ldm) {

        //Create shared memory.
        __shared__ typename Traits::SharedStorage shared_storage;

        // Create those iterators.
        typedef typename Traits::GlobalLoadStreamB GlobalLoadStreamB;
        typename GlobalLoadStreamB::Params global_load_params;
        GemmDesc desc(M, N, K);
        global_load_params.initialize(desc, input, ldm);

        GlobalLoadStreamB stream_b(global_load_params, shared_storage.main_loop.stream_b.global, M, N, K, cutlass::make_Coord(0, 0, 0));
        stream_b.copy();
        stream_b.commit();

        // store barrier
        __syncthreads();

        // one thread writes everything out
        if (threadIdx.x == 0) {
            for (int i = 0; i < M*K; ++i) {
                output[i] = shared_storage.main_loop.stream_b.shared[i];
            }
        }

    }

////////////////////////////////////////////////////////////////////////////////////////////////////
    template <
        typename CtaTile,                                  // concept: Shape
        typename DestType,                                 // raw data type
        typename SourceType                                // raw data type
    >
    class VerifyDataMovement {
        public:

        /// Tensor to store the destination data
        cutlass::HostTensor<DestType> destination;

        /// Tensor to store the source data
        cutlass::HostTensor<SourceType> source;

        /// Verification utility
        typedef test::VerifyLayout<
            DestType,
            test::CoordinatePack<DestType>,
            SourceType,
            test::CoordinatePack<SourceType> > VerifyLayout;

        /// Verification object
        VerifyLayout verify_layout;

        public:

        VerifyDataMovement() { }

        VerifyDataMovement(test::Layout const &source_layout) {

        // Actual layout here doesn't matter here, just the number of elements
        destination.resize_matrix(CtaTile::kH, CtaTile::kW, cutlass::MatrixLayout::kRowMajor);
        source.resize_matrix(CtaTile::kH, CtaTile::kW, cutlass::MatrixLayout::kRowMajor);

        verify_layout.initialize(source, source_layout);
        destination.fill(0);

        destination.sync_device();
        source.sync_device();
    }

        /// Verifies resulting layout
        bool verify(test::Layout const & destination_layout) {

            destination.sync_host();

            typename VerifyLayout::VisitorVerbose visitor(std::cout);

            bool passed = verify_layout.verify(
                destination,
                destination_layout,
                visitor);

            return passed;
        }
    };


////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(Gemm_shared_tile, A_float_contiguous) {

    static int const M = 64;
    static int const N = 64;
    static int const K = 8;

    typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<K, N, M> >
      SgemmTraits;

    typedef test::Layout::Span Span;
    test::Layout::SpanVector dst_layout;
    test::Layout::SpanVector src_layout;

    // define the source layout
    src_layout.push_back(Span(0, K));
    src_layout.push_back(Span(1, M));

    typedef VerifyDataMovement<
        cutlass::Shape<1, M, K, 1>,
        float,
        float
    > VerifyDataMovement_t;

    VerifyDataMovement_t testbed(src_layout);


    test::Gemm_load_global_store_shared_a< SgemmTraits ><<<
        dim3(1,1,1),
        dim3(SgemmTraits::kThreads, 1)
    >>>(
        testbed.destination.device_data(),
        testbed.source.device_data(),
        M,
        N,
        K,
        M
    );

    cudaError_t result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

    // define the destination layout
    dst_layout.push_back(Span(0, K));
    dst_layout.push_back(Span(1, M));

    EXPECT_TRUE(testbed.verify(dst_layout));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(Gemm_shared_tile, A_double_contiguous) {

    static int const M = 64;
    static int const N = 64;
    static int const K = 8;

    typedef cutlass::gemm::DgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<K, N, M> >
      DgemmTraits;

    typedef test::Layout::Span Span;
    test::Layout::SpanVector dst_layout;
    test::Layout::SpanVector src_layout;

    // define the source layout
    src_layout.push_back(Span(0, K));
    src_layout.push_back(Span(1, M));

    typedef VerifyDataMovement<
        cutlass::Shape<1, M, K, 1>,
        double,
        double
    > VerifyDataMovement_t;

    VerifyDataMovement_t testbed(src_layout);

    test::Gemm_load_global_store_shared_a< DgemmTraits ><<<
        dim3(1,1,1),
        dim3(DgemmTraits::kThreads, 1)
    >>>(
        testbed.destination.device_data(),
        testbed.source.device_data(),
        M,
        N,
        K,
        M
    );

    cudaError_t result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

    // define the destination layout
    dst_layout.push_back(Span(0, K));
    dst_layout.push_back(Span(1, M));

    EXPECT_TRUE(testbed.verify(dst_layout));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(Gemm_shared_tile, B_float_contiguous) {

    static int const M = 64;
    static int const N = 64;
    static int const K = 8;

    typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<K, N, M> >
      SgemmTraits;

    typedef test::Layout::Span Span;
    test::Layout::SpanVector dst_layout;
    test::Layout::SpanVector src_layout;

    // define the source layout
    src_layout.push_back(Span(0, K));
    src_layout.push_back(Span(1, M));

    typedef VerifyDataMovement<
        cutlass::Shape<1, M, K, 1>,
        float,
        float
    > VerifyDataMovement_t;

    VerifyDataMovement_t testbed(src_layout);


    test::Gemm_load_global_store_shared_b< SgemmTraits ><<<
        dim3(1,1,1),
        dim3(SgemmTraits::kThreads, 1)
    >>>(
        testbed.destination.device_data(),
        testbed.source.device_data(),
        M,
        N,
        K,
        M
    );

    cudaError_t result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

    // define the destination layout
    dst_layout.push_back(Span(0, K));
    dst_layout.push_back(Span(1, M));

    EXPECT_TRUE(testbed.verify(dst_layout));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(Gemm_shared_tile, B_double_contiguous) {

    static int const M = 64;
    static int const N = 64;
    static int const K = 8;


    typedef cutlass::gemm::DgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<K, N, M> >
      DgemmTraits;

    typedef test::Layout::Span Span;
    test::Layout::SpanVector dst_layout;
    test::Layout::SpanVector src_layout;

    // define the source layout
    src_layout.push_back(Span(0, K));
    src_layout.push_back(Span(1, M));

    typedef VerifyDataMovement<
        cutlass::Shape<1, M, K, 1>,
        double,
        double
    > VerifyDataMovement_t;

    VerifyDataMovement_t testbed(src_layout);

    test::Gemm_load_global_store_shared_b< DgemmTraits ><<<
        dim3(1,1,1),
        dim3(DgemmTraits::kThreads, 1)
    >>>(
        testbed.destination.device_data(),
        testbed.source.device_data(),
        M,
        N,
        K,
        M
    );

    cudaError_t result = cudaDeviceSynchronize();
    ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

    // define the destination layout
    dst_layout.push_back(Span(0, K));
    dst_layout.push_back(Span(1, M));

    EXPECT_TRUE(testbed.verify(dst_layout));
}
////////////////////////////////////////////////////////////////////////////////////////////////////

}

