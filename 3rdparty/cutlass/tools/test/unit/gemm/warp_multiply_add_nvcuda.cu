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

////////////////////////////////////////////////////////////////////////////////////////////////////

// Guard conditions around the entire file.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass_unit_tests.h"
#include "tools/util/half.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "cutlass/gemm/warp_multiply_add_nvcuda.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tests for Warp-level Multiply Add operation using NvCuda API
//

namespace test {

///
template <typename WarpMultiplyAdd>
__global__ void warp_multiply_add(
    typename WarpMultiplyAdd::ScalarA const *A,
    int lda,
    typename WarpMultiplyAdd::ScalarB const *B,
    int ldb,
    typename WarpMultiplyAdd::ScalarC *C, int ldc) {

    typename WarpMultiplyAdd::LoadIteratorA iteratorA(A, lda);
    typename WarpMultiplyAdd::LoadIteratorB iteratorB(B, ldb);
    typename WarpMultiplyAdd::StoreIteratorC iteratorC(C, ldc);

    typename WarpMultiplyAdd::FragmentA fragmentA;
    typename WarpMultiplyAdd::FragmentB fragmentB;
    typename WarpMultiplyAdd::FragmentC fragmentC;

    iteratorA.load(fragmentA);
    iteratorB.load(fragmentB);

    fragmentC.clear();

    WarpMultiplyAdd::multiply_add(fragmentC, fragmentA, fragmentB, fragmentC);

    iteratorC.store(fragmentC);
}

/// Test environment for Warp Multiply Add operation
template <
    cutlass::MatrixLayout::Kind LayoutA,
    cutlass::MatrixLayout::Kind LayoutB,
    cutlass::MatrixLayout::Kind LayoutC,
    typename ScalarC,
    typename WarpTile,
    typename WmmaTile
>
struct TestWarpMultiplyAdd {

    typedef cutlass::gemm::WarpMultiplyAddNvcuda<
        LayoutA,
        LayoutB,
        LayoutC,
        half,
        half,
        ScalarC,
        WarpTile,
        cutlass::Shape<1, 1, 1, 1>,
        WmmaTile
    > WarpMultiplyAdd;

    /// Testbed type
    typedef test::GemmTestbed<
        cutlass::half_t,
        cutlass::half_t,
        ScalarC,
        ScalarC,
        ScalarC
    > GemmTestbed;

    //
    // Data members
    //

    GemmTestbed testbed;

    //
    // Methods
    //

    TestWarpMultiplyAdd(): testbed(
        WarpTile::kW,   // M
        WarpTile::kH,   // N
        WarpTile::kD,   // K
        cutlass::convert(LayoutA),
        cutlass::convert(LayoutB),
        1,
        0,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        cutlass::convert(LayoutC))
    {

    }

    /// Run
    void run() {
        testbed.initialize();

        // launch
        warp_multiply_add<WarpMultiplyAdd><<<
            dim3(1,1,1), dim3(32, 1, 1)
        >>>(
            testbed.ptr_A(),
            testbed.lda(),
            testbed.ptr_B(),
            testbed.ldb(),
            testbed.ptr_computed(),
            testbed.ldc()
        );

        // verify
        ASSERT_TRUE(testbed.verify_with_host());
    }
};

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename ScalarC,
    typename WarpTile,
    typename WmmaTile
>
struct TestWarpMultiplyAddForAllLayouts {

    void run() {

        test::TestWarpMultiplyAdd<
            cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kColumnMajor,
            ScalarC,
            WarpTile,
            WmmaTile
        >().run();

        test::TestWarpMultiplyAdd<
            cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kColumnMajor,
            ScalarC,
            WarpTile,
            WmmaTile
        >().run();

        test::TestWarpMultiplyAdd<
            cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kColumnMajor,
            ScalarC,
            WarpTile,
            WmmaTile
        >().run();

        test::TestWarpMultiplyAdd<
            cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kColumnMajor,
            ScalarC,
            WarpTile,
            WmmaTile
        >().run();


        test::TestWarpMultiplyAdd<
            cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kRowMajor,
            ScalarC,
            WarpTile,
            WmmaTile
        >().run();

        test::TestWarpMultiplyAdd<
            cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kRowMajor,
            ScalarC,
            WarpTile,
            WmmaTile
        >().run();

        test::TestWarpMultiplyAdd<
            cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kRowMajor,
            ScalarC,
            WarpTile,
            WmmaTile
        >().run();

        test::TestWarpMultiplyAdd<
            cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kRowMajor,
            ScalarC,
            WarpTile,
            WmmaTile
        >().run();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 16x16x16 WMMA Tile Shape with F32 accumulation
//

TEST(WmmaGemm, WarpMultiplyAdd_f32_16x16x16_16x16x16) {
    TestWarpMultiplyAddForAllLayouts<
        float,
        cutlass::Shape<16, 16, 16>,
        cutlass::Shape<16, 16, 16>
    >().run();
}

TEST(WmmaGemm, WarpMultiplyAdd_f32_16x16x32_16x16x16) {
    TestWarpMultiplyAddForAllLayouts<
        float,
        cutlass::Shape<16, 16, 32>,
        cutlass::Shape<16, 16, 16>
    >().run();
}

TEST(WmmaGemm, WarpMultiplyAdd_f32_16x32x32_16x16x16) {
    TestWarpMultiplyAddForAllLayouts<
        float,
        cutlass::Shape<16, 32, 32>,
        cutlass::Shape<16, 16, 16>
    >().run();
}

TEST(WmmaGemm, WarpMultiplyAdd_f32_16x32x64_16x16x16) {
    TestWarpMultiplyAddForAllLayouts<
        float,
        cutlass::Shape<16, 32, 64>,
        cutlass::Shape<16, 16, 16>
    >().run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
