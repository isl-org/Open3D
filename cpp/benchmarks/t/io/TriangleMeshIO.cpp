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

#include "open3d/io/TriangleMeshIO.h"

#include <benchmark/benchmark.h>

#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/t/io/TriangleMeshIO.h"

namespace open3d {
namespace t {
namespace geometry {

data::KnotMesh knot_data;

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

BENCHMARK_CAPTURE(IOReadLegacyTriangleMesh, CPU, knot_data.GetPath())
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorTriangleMesh, CPU, knot_data.GetPath())
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
