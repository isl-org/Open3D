// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include "open3d/data/Dataset.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/TriangleMeshIO.h"

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
