// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <benchmark/benchmark.h>

#include "open3d/data/Dataset.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/DataManager.h"

namespace open3d {
namespace benchmarks {

class SamplePointsFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) {
        data::KnotMesh knot_data;
        trimesh = io::CreateMeshFromFile(knot_data.GetPath());
    }

    void TearDown(const benchmark::State& state) {
        // empty
    }
    std::shared_ptr<geometry::TriangleMesh> trimesh;
};

BENCHMARK_DEFINE_F(SamplePointsFixture, Poisson)(benchmark::State& state) {
    for (auto _ : state) {
        trimesh->SamplePointsPoissonDisk(state.range(0));
    }
}

BENCHMARK_REGISTER_F(SamplePointsFixture, Poisson)->Args({123})->Args({1000});

BENCHMARK_DEFINE_F(SamplePointsFixture, Uniform)(benchmark::State& state) {
    for (auto _ : state) {
        trimesh->SamplePointsUniformly(state.range(0));
    }
}

BENCHMARK_REGISTER_F(SamplePointsFixture, Uniform)->Args({123})->Args({1000});

}  // namespace benchmarks
}  // namespace open3d
