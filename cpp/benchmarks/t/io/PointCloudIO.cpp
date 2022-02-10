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
#include "open3d/data/Dataset.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/DataManager.h"

namespace open3d {
namespace t {
namespace geometry {

// The `Read` benchmark functions are dependent on the corresponding `Write`
// benchmark functions to generate the point cloud file in the required format.
// To run benchmarks in this file, run the following command from inside the
// build directory: ./bin/benchmarks --benchmark_filter=".*IO.*"

// This file is just used to load the `point cloud` data. So, format of this
// file is not important.
data::PLYPointCloud pointcloud_ply_data;
static const std::string input_path_pcd = pointcloud_ply_data.GetPath();

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

#define ENUM_BM_IO_EXTENSION_FORMAT(EXTENSION_NAME, FILE_NAME, FORMAT_NAME,    \
                                    ASCII, COMPRESSED)                         \
    BENCHMARK_CAPTURE(IOWriteLegacyPointCloud, EXTENSION_NAME##_##FORMAT_NAME, \
                      input_path_pcd,                                          \
                      std::string("tensor_") + std::string(FILE_NAME), ASCII,  \
                      COMPRESSED)                                              \
            ->Unit(benchmark::kMillisecond);                                   \
    BENCHMARK_CAPTURE(IOReadLegacyPointCloud, EXTENSION_NAME##_##FORMAT_NAME,  \
                      std::string("tensor_") + std::string(FILE_NAME))         \
            ->Unit(benchmark::kMillisecond);                                   \
    BENCHMARK_CAPTURE(IOWriteTensorPointCloud, EXTENSION_NAME##_##FORMAT_NAME, \
                      input_path_pcd,                                          \
                      std::string("legacy_") + std::string(FILE_NAME), ASCII,  \
                      COMPRESSED)                                              \
            ->Unit(benchmark::kMillisecond);                                   \
    BENCHMARK_CAPTURE(IOReadTensorPointCloud, EXTENSION_NAME##_##FORMAT_NAME,  \
                      std::string("legacy_") + std::string(FILE_NAME))         \
            ->Unit(benchmark::kMillisecond);

#define ENUM_BM_IO_EXTENSION(EXTENSION_NAME, EXTENSION)                        \
    ENUM_BM_IO_EXTENSION_FORMAT(                                               \
            EXTENSION_NAME, std::string("pcd_ascii") + std::string(EXTENSION), \
            ASCII_UNCOMPRESSED, true, false)                                   \
    ENUM_BM_IO_EXTENSION_FORMAT(                                               \
            EXTENSION_NAME, std::string("pcd_bin") + std::string(EXTENSION),   \
            BINARY_UNCOMPRESSED, false, false)                                 \
    ENUM_BM_IO_EXTENSION_FORMAT(                                               \
            EXTENSION_NAME,                                                    \
            std::string("pcd_bin_compressed") + std::string(EXTENSION),        \
            BINARY_COMPRESSED, false, true)

ENUM_BM_IO_EXTENSION(PCD, ".pcd")
ENUM_BM_IO_EXTENSION(PLY, ".ply")
ENUM_BM_IO_EXTENSION(PTS, ".pts")

}  // namespace geometry
}  // namespace t
}  // namespace open3d
