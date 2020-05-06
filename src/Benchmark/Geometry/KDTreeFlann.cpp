// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "benchmark/benchmark.h"

using namespace Eigen;
using namespace open3d;
using namespace std;

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

        Vector3d query = {(pos_ + 0.1) * step, 0., 0.};
        double radius = radiusInSteps * step;
        vector<int> indices;
        vector<double> distance2;

        int result = kdtree_.SearchRadius<Vector3d>(query, radius, indices,
                                                    distance2);
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
