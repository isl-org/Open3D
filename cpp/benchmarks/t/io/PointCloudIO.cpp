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

static const std::string path_pcd =
        std::string(TEST_DATA_DIR) + "/fragment.pcd";
static const std::string path_ply = "/home/rey/bedroom.ply";

void ReadTensorPointCloud(benchmark::State& state,
                          const std::string& file_path) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});

    for (auto _ : state) {
        t::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});
    }
}

void ReadLegacyPointCloud(benchmark::State& state,
                          const std::string& file_path) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});

    for (auto _ : state) {
        open3d::io::ReadPointCloud(file_path, pcd,
                                   {"auto", false, false, false});
    }
}

void WriteTensorPointCloud(benchmark::State& state,
                           const std::string& file_path) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});

    t::io::WritePointCloud("/home/rey/t_pcd_0.ply", pcd);

    t::geometry::PointCloud pcd_saved;
    t::io::ReadPointCloud("/home/rey/t_pcd_0.ply", pcd_saved, {"auto", false, false, false});

    std::cout << " Points Same ? "
              << pcd.GetPointAttr("points").AllClose(
                         pcd_saved.GetPointAttr("points"))
              << std::endl;
    std::cout << " Normals Same ? "
              << pcd.GetPointAttr("normals").AllClose(
                         pcd_saved.GetPointAttr("normals"))
              << std::endl;
    std::cout << " Colors Same ? "
              << pcd.GetPointAttr("colors").AllClose(
                         pcd_saved.GetPointAttr("colors"))
              << std::endl;

    int i = 0;
    for (auto _ : state) {
        std::string filename_loop =
                "/home/rey/t_pcd_" + std::to_string(++i) + ".ply";
        t::io::WritePointCloud(filename_loop, pcd);
    }
}

void WriteLegacyPointCloud(benchmark::State& state,
                           const std::string& file_path) {
    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloud(file_path, pcd, {"auto", false, false, false});

    open3d::io::WritePointCloud("/home/rey/l_pcd_0.ply", pcd);
    int i = 0;
    for (auto _ : state) {
        std::string filename_loop =
                "/home/rey/l_pcd_" + std::to_string(++i) + ".ply";
        open3d::io::WritePointCloud(filename_loop, pcd);
    }
}

BENCHMARK_CAPTURE(ReadLegacyPointCloud, PLY, path_ply)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadTensorPointCloud, PLY, path_ply)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadLegacyPointCloud, PCD, path_pcd)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ReadTensorPointCloud, PCD, path_pcd)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteLegacyPointCloud, PLY, path_ply)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(WriteTensorPointCloud, PLY, path_ply)
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace open3d
