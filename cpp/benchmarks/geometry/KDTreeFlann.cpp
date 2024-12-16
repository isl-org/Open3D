// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/KDTreeFlann.h"

#include <benchmark/benchmark.h>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace benchmarks {

class TestKDTreeLine0 {
    constexpr static double step = .139;
    geometry::PointCloud pc_;
    geometry::KDTreeFlann kdtree_;

    int pos_ = 0;
    int size_ = 0;

public:
    void setup(int size) {
        if (this->size_ == size) return;
        utility::LogInfo("setup KDTree size={:d}", size);
        pc_.Clear();

        this->size_ = size;
        for (int i = 0; i < size; ++i) {
            pc_.points_.push_back({double(i) * step, 0., 0.});
        }

        kdtree_.SetGeometry(pc_);
        pos_ = size / 2;
    }

    void search(int radiusInSteps) {
        pos_ += 2 * radiusInSteps;
        if (pos_ >= size_ - radiusInSteps) {
            pos_ = radiusInSteps;
        }

        Eigen::Vector3d query = {(pos_ + 0.1) * step, 0., 0.};
        double radius = radiusInSteps * step;
        std::vector<int> indices;
        std::vector<double> distance2;

        int result = kdtree_.SearchRadius<Eigen::Vector3d>(query, radius,
                                                           indices, distance2);
        if (result != radiusInSteps * 2) {
            utility::LogError("size={:d} radiusInSteps={:d} pos={:d} num={:d}",
                              size_, radiusInSteps, pos_, result);
        }
    }
};
// reuse the same instance so we don't recreate the kdtree every time
TestKDTreeLine0 testKDTreeLine0;

static void BM_TestKDTreeLine0(benchmark::State& state) {
    // state.range(n) are arguments that are passed to us
    int radius = state.range(0);
    int size = state.range(1);
    testKDTreeLine0.setup(size);
    for (auto _ : state) {
        testKDTreeLine0.search(radius);
    }
}
// a few specific sized tests, each ->Args({params}) will be a test
BENCHMARK(BM_TestKDTreeLine0)->Args({1 << 5, 1 << 10})->Args({1 << 9, 1 << 11});
// let benchmark vary parameters for us; run each test only for 0.1sec so it
// doesn't take too long
BENCHMARK(BM_TestKDTreeLine0)
        ->MinTime(0.1)
        ->Ranges({{1 << 0, 1 << 14}, {1 << 16, 1 << 22}});

}  // namespace benchmarks
}  // namespace open3d
