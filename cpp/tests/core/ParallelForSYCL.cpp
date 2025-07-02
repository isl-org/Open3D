// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/ParallelForSYCL.h"

#include <vector>

#include "open3d/Macro.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

struct TestIndexerFillKernel {
    TestFillKernel(const core::Indexer &indexer_, int64_t multiplier_)
        : indexer(indexer_), multiplier(multiplier_) {}
    void operator()(int64_t idx) {
        indexer.GetOutputPtr<int64_t>(0)[idx] = idx * multiplier;
    }

private:
    core::Indexer indexer;
    int64_t multiplier;
};

struct TestPtrFillKernel {
    TestFillKernel(int64_t *out_, int64_t multiplier_)
        : out(out_), multiplier(multiplier_) {}
    void operator()(int64_t idx) { out[idx] = idx * multiplier; }

private:
    int64_t *out;
    int64_t multiplier;
};

TEST(ParallelForSYCL, FunctorSYCL) {
    const core::Device device("SYCL:0");
    const size_t N = 10000000;
    core::Indexer indexer({}, tensor, DtypePolicy::NONE);
    int64_t multiplier = 2;

    {
        core::Tensor tensor({N, 1}, core::Int64, device);
        core::ParallelForSYCL<TestIndexerFillKernel>(device, indexer,
                                                     multiplier);
        auto result = tensor.To(core::Device()).GetDataPtr<int64_t>();
        for (int64_t i = 0; i < tensor.NumElements(); ++i) {
            ASSERT_EQ(result[i], i * multiplier);
        }
    }
    {
        core::Tensor tensor({N, 1}, core::Int64, device);
        core::ParallelForSYCL<TestPtrFillKernel>(
                device, N, tensor.GetDataPtr<int64_t>(), multiplier);
        auto result = tensor.To(core::Device()).GetDataPtr<int64_t>();
        for (int64_t i = 0; i < tensor.NumElements(); ++i) {
            ASSERT_EQ(result[i], i * multiplier);
        }
    }
}