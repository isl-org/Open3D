// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/TriangleMeshIO.h"

#include <benchmark/benchmark.h>

#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/t/io/TriangleMeshIO.h"

namespace open3d {
namespace t {
namespace geometry {

static const std::string& KnotMeshPath() {
    static data::KnotMesh knot_data;
    static const std::string path = knot_data.GetPath();
    return path;
}

void IOReadLegacyTriangleMesh(benchmark::State& state,
                              const std::string& input_file_path) {
    open3d::geometry::TriangleMesh mesh;
    open3d::io::ReadTriangleMesh(input_file_path, mesh);

    for (auto _ : state) {
        open3d::io::ReadTriangleMesh(input_file_path, mesh);
    }
}

void IOReadTensorTriangleMesh(benchmark::State& state,
                              const std::string& input_file_path) {
    t::geometry::TriangleMesh mesh;
    t::io::ReadTriangleMesh(input_file_path, mesh);

    for (auto _ : state) {
        t::io::ReadTriangleMesh(input_file_path, mesh);
    }
}

BENCHMARK_CAPTURE(IOReadLegacyTriangleMesh, CPU, KnotMeshPath())
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorTriangleMesh, CPU, KnotMeshPath())
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
