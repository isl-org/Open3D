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

#include "open3d/io/PointCloudIO.h"

#include <benchmark/benchmark.h>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"

namespace open3d {
namespace t {
namespace geometry {

// The `Read` benchmark functions are dependent on the corresponding `Write`
// benchmark functions to generate the point cloud file in the required format.
// To run benchmarks in this file, run the following command from inside the
// build directory: ./bin/benchmarks --benchmark_filter=".*IO.*"

// This file is just used to load the `point cloud` data. So, format of this
// file is not important.
static const std::string input_path_pcd =
        std::string(TEST_DATA_DIR) + "/fragment.ply";

void IOWriteLegacyPointCloud(benchmark::State& state,
                             const std::string& input_file_path,
                             const std::string& output_file_path,
                             const bool write_ascii,
                             const bool write_compressed) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(input_file_path, pcd,
                               {"auto", false, false, false});

    open3d::io::WritePointCloud(
            output_file_path, pcd,
            open3d::io::WritePointCloudOption(write_ascii, write_compressed,
                                              false, {}));

    for (auto _ : state) {
        open3d::io::WritePointCloud(
                output_file_path, pcd,
                open3d::io::WritePointCloudOption(write_ascii, write_compressed,
                                                  false, {}));
    }
}

void IOReadLegacyPointCloud(benchmark::State& state,
                            const std::string& input_file_path) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(input_file_path, pcd,
                               {"auto", false, false, false});

    for (auto _ : state) {
        open3d::io::ReadPointCloud(input_file_path, pcd,
                                   {"auto", false, false, false});
    }
}

void IOWriteTensorPointCloud(benchmark::State& state,
                             const std::string& input_file_path,
                             const std::string& output_file_path,
                             const bool write_ascii,
                             const bool write_compressed) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(input_file_path, pcd, {"auto", false, false, false});

    t::io::WritePointCloud(output_file_path, pcd,
                           open3d::io::WritePointCloudOption(
                                   write_ascii, write_compressed, false, {}));

    for (auto _ : state) {
        t::io::WritePointCloud(
                output_file_path, pcd,
                open3d::io::WritePointCloudOption(write_ascii, write_compressed,
                                                  false, {}));
    }
}

void IOReadTensorPointCloud(benchmark::State& state,
                            const std::string& input_file_path) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(input_file_path, pcd, {"auto", false, false, false});

    for (auto _ : state) {
        t::io::ReadPointCloud(input_file_path, pcd,
                              {"auto", false, false, false});
    }
}

BENCHMARK_CAPTURE(IOWriteLegacyPointCloud,
                  PCD ASCII UNCOMPRESSED,
                  input_path_pcd,
                  "test_lpcd_ascii.pcd",
                  true,
                  false)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadLegacyPointCloud,
                  PCD ASCII UNCOMPRESSED,
                  "test_lpcd_ascii.pcd")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteTensorPointCloud,
                  PCD ASCII UNCOMPRESSED,
                  input_path_pcd,
                  "test_tpcd_ascii.pcd",
                  true,
                  false)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorPointCloud,
                  PCD ASCII UNCOMPRESSED,
                  "test_tpcd_ascii.pcd")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteLegacyPointCloud,
                  PCD BINARY UNCOMPRESSED,
                  input_path_pcd,
                  "test_lpcd_bin.pcd",
                  false,
                  false)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadLegacyPointCloud,
                  PCD BINARY UNCOMPRESSED,
                  "test_lpcd_bin.pcd")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteTensorPointCloud,
                  PCD BINARY UNCOMPRESSED,
                  input_path_pcd,
                  "test_tpcd_bin.pcd",
                  false,
                  false)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorPointCloud,
                  PCD BINARY UNCOMPRESSED,
                  "test_tpcd_bin.pcd")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteLegacyPointCloud,
                  PCD BINARY COMPRESSED,
                  input_path_pcd,
                  "test_lpcd_bin_compressed.pcd",
                  false,
                  true)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadLegacyPointCloud,
                  PCD BINARY COMPRESSED,
                  "test_lpcd_bin_compressed.pcd")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteTensorPointCloud,
                  PCD BINARY COMPRESSED,
                  input_path_pcd,
                  "test_tpcd_bin_compressed.pcd",
                  false,
                  true)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorPointCloud,
                  PCD BINARY COMPRESSED,
                  "test_tpcd_bin_compressed.pcd")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteLegacyPointCloud,
                  PLY ASCII UNCOMPRESSED,
                  input_path_pcd,
                  "test_lpcd_ascii.ply",
                  true,
                  false)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadLegacyPointCloud,
                  PLY ASCII UNCOMPRESSED,
                  "test_lpcd_ascii.ply")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteTensorPointCloud,
                  PLY ASCII UNCOMPRESSED,
                  input_path_pcd,
                  "test_tpcd_ascii.ply",
                  true,
                  false)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorPointCloud,
                  PLY ASCII UNCOMPRESSED,
                  "test_tpcd_ascii.ply")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteLegacyPointCloud,
                  PLY BINARY UNCOMPRESSED,
                  input_path_pcd,
                  "test_lpcd_bin.ply",
                  false,
                  false)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadLegacyPointCloud,
                  PLY BINARY UNCOMPRESSED,
                  "test_lpcd_bin.ply")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteTensorPointCloud,
                  PLY BINARY UNCOMPRESSED,
                  input_path_pcd,
                  "test_tpcd_bin.ply",
                  false,
                  false)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorPointCloud,
                  PLY BINARY UNCOMPRESSED,
                  "test_tpcd_bin.ply")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteLegacyPointCloud,
                  PLY BINARY COMPRESSED,
                  input_path_pcd,
                  "test_lpcd_bin_compressed.ply",
                  false,
                  true)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadLegacyPointCloud,
                  PLY BINARY COMPRESSED,
                  "test_lpcd_bin_compressed.ply")
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOWriteTensorPointCloud,
                  PLY BINARY COMPRESSED,
                  input_path_pcd,
                  "test_tpcd_bin_compressed.ply",
                  false,
                  true)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorPointCloud,
                  PLY BINARY COMPRESSED,
                  "test_tpcd_bin_compressed.ply")
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
