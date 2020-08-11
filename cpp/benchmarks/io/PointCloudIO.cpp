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

#include "open3d/io/PointCloudIO.h"

#include <benchmark/benchmark.h>

#include "open3d/utility/Console.h"

namespace open3d {
namespace benchmarks {

using open3d::io::ReadPointCloud;
using open3d::io::WritePointCloud;

namespace {

template <class T>
double AverageDistance(const std::vector<T> &a, const std::vector<T> &b) {
    if (a.size() != b.size()) {
        utility::LogError("vectors different size {} {}", a.size(), b.size());
    }
    if (a.size() == 0) {
        return 0.;
    }
    double total = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        total += (a[i] - b[i]).norm();
    }
    return total / a.size();
}

enum class IsAscii : bool { BINARY = false, ASCII = true };
enum class Compressed : bool { UNCOMPRESSED = false, COMPRESSED = true };
enum class Compare : uint32_t {
    // Points are always compared
    NONE = 0,
    NORMALS = 1 << 0,
    COLORS = 1 << 1,
    NORMALS_AND_COLORS = NORMALS | COLORS
};
struct ReadWritePCArgs {
    std::string filename;
    IsAscii write_ascii;
    Compressed compressed;
    Compare compare;
};
std::vector<ReadWritePCArgs> g_pc_args({
        // PCD has ASCII, BINARY, and BINARY_COMPRESSED
        {"testau.pcd", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 0
        {"testbu.pcd", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 1
        {"testbc.pcd", IsAscii::BINARY, Compressed::COMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 2
        {"testb.ply", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 3
        {"testa.ply", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 4
        {"test.pts", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::COLORS},  // 5
        {"test.xyz", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NONE},  // 6
        {"test.xyzn", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NORMALS},  // 7
        {"test.xyzrgb", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::COLORS},  // 8
});

class TestPCGrid0 {
    geometry::PointCloud pc_;
    int size_ = 0;
    const bool print_progress = false;

public:
    void Setup(int size) {
        if (size_ == size) return;
        utility::LogInfo("setup PCGrid size={}", size);
        pc_.Clear();

        size_ = size;
        for (int i = 0; i < size; ++i) {
            // provide somewhat random numbers everywhere, so compression
            // doesn't get a free pass
            pc_.points_.push_back({std::sin(i * .8969920581) * 1000.,
                                   std::sin(i * .3898546778) * 1000.,
                                   std::sin(i * .2509962463) * 1000.});
            pc_.normals_.push_back({std::sin(i * .4472367685),
                                    std::sin(i * .9698787116),
                                    std::sin(i * .7072878517)});
            // color needs to be [0,1]
            pc_.colors_.push_back({std::fmod(i * .4241490710, 1.0),
                                   std::fmod(i * .6468026221, 1.0),
                                   std::fmod(i * .5376722873, 1.0)});
        }
    }

    void WriteRead(int pc_args_id) {
        const auto &args = g_pc_args[pc_args_id];
        const auto &pc = pc_;
        // we loose some precision when saving generated data
        if (!WritePointCloud(args.filename, pc,
                             {bool(args.write_ascii), bool(args.compressed),
                              print_progress})) {
            utility::LogError("Failed to write to {}", args.filename);
        }
        geometry::PointCloud pc2;
        if (!ReadPointCloud(args.filename, pc2,
                            {"auto", false, false, print_progress})) {
            utility::LogError("Failed to read from {}", args.filename);
        }
        auto CheckLE = [](double a, double b) {
            if (a <= b) return;
            utility::LogError("Error too high: {} {}", a, b);
        };

        const double pointsMaxError =
                1e-3;  //.ply ascii has the highest error, others <1e-4
        CheckLE(AverageDistance(pc.points_, pc2.points_), pointsMaxError);
        if (int(args.compare) & int(Compare::NORMALS)) {
            const double normalsMaxError =
                    1e-6;  //.ply ascii has the highest error, others <1e-7
            CheckLE(AverageDistance(pc.normals_, pc2.normals_),
                    normalsMaxError);
        }
        if (int(args.compare) & int(Compare::COLORS)) {
            const double colorsMaxError =
                    1e-2;  // colors are saved as uint8_t[3] in a lot of formats
            CheckLE(AverageDistance(pc.colors_, pc2.colors_), colorsMaxError);
        }
    }
};
// reuse the same instance so we don't recreate the point cloud every time
TestPCGrid0 test_pc_grid0;

}  // namespace

static void BM_TestPCGrid0(::benchmark::State &state) {
    // state.range(n) are arguments that are passed to us
    int pc_args_id = state.range(0);
    int size = state.range(1);
    test_pc_grid0.Setup(size);
    for (auto _ : state) {
        test_pc_grid0.WriteRead(pc_args_id);
    }
}
static void BM_TestPCGrid0_Args(benchmark::internal::Benchmark *b) {
    for (int j = 4 * 1024; j <= 256 * 1024; j *= 8) {
        for (int i = 0; i < int(g_pc_args.size()); ++i) {
            b->Args({i, j});
        }
    }
}

BENCHMARK(BM_TestPCGrid0)->MinTime(0.1)->Apply(BM_TestPCGrid0_Args);

}  // namespace benchmarks
}  // namespace open3d
