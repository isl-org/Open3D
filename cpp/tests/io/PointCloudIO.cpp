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

#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

using open3d::io::ReadPointCloud;
using open3d::io::ReadPointCloudOption;
using open3d::io::WritePointCloud;
using open3d::io::WritePointCloudOption;

namespace {

template <class T>
double MaxDistance(const std::vector<T> &a, const std::vector<T> &b) {
    // Note: cannot use ASSERT_EQ because we return non-void
    EXPECT_EQ(a.size(), b.size());
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, (a[i] - b[i]).norm());
    }
    return m;
}

void RandPC(geometry::PointCloud &pc, int size = 100) {
    Eigen::Vector3d one(1, 1, 1);

    pc.points_.resize(size);
    pc.normals_.resize(size);
    pc.colors_.resize(size);

    Rand(pc.points_, one * -1000, one * 1000, 0);
    Rand(pc.normals_, one * -1, one, 0);
    // Rand*255 seems to give whole numbers, test non-whole numbers for colors
    Rand(pc.colors_, one * 0, one * .9973143, 0);
}

}  // namespace

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
std::vector<ReadWritePCArgs> pcArgs({
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
                            // test subsets of PCD
        {"testaup.pcd", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::NONE},  // 9
        {"testbup.pcd", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NONE},  // 10
        {"testbcp.pcd", IsAscii::BINARY, Compressed::COMPRESSED,
         Compare::NONE},  // 11
        {"testaun.pcd", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::NORMALS},  // 12
        {"testbun.pcd", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NORMALS},  // 13
        {"testbcn.pcd", IsAscii::BINARY, Compressed::COMPRESSED,
         Compare::NORMALS},  // 14
        {"testauc.pcd", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::COLORS},  // 15
        {"testbuc.pcd", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::COLORS},  // 16
        {"testbcc.pcd", IsAscii::BINARY, Compressed::COMPRESSED,
         Compare::COLORS},  // 17
                            // test subsets of PLY
        {"testbp.ply", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NONE},  // 18
        {"testap.ply", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::NONE},  // 19
        {"testbn.ply", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NORMALS},  // 20
        {"testan.ply", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::NORMALS},  // 21
        {"testbc.ply", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::COLORS},  // 22
        {"testac.ply", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::COLORS},  // 23
                            // test subsets of PTS
        {"testp.pts", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NONE},  // 24
});

class ReadWritePC : public testing::TestWithParam<ReadWritePCArgs> {};
INSTANTIATE_TEST_SUITE_P(ReadWritePC, ReadWritePC, testing::ValuesIn(pcArgs));

TEST_P(ReadWritePC, Basic) {
    ReadWritePCArgs args = GetParam();
    geometry::PointCloud pc;
    RandPC(pc);

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();

    // we loose some precision when saving generated data
    // test writing if we have point, normal, and colors in pc
    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc2;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc2,
                               {"auto", false, false, true}));
    const double points_max_error =
            1e-3;  //.ply ascii has the highest error, others <1e-4
    EXPECT_LT(MaxDistance(pc.points_, pc2.points_), points_max_error);
    if (int(args.compare) & int(Compare::NORMALS)) {
        SCOPED_TRACE("Normals");
        const double normals_max_error =
                1e-6;  //.ply ascii has the highest error, others <1e-7
        EXPECT_LT(MaxDistance(pc.normals_, pc2.normals_), normals_max_error);
    }
    if (int(args.compare) & int(Compare::COLORS)) {
        SCOPED_TRACE("Colors");
        const double colors_max_error =
                1e-2;  // colors are saved as uint8_t[3] in a lot of formats
        EXPECT_LT(MaxDistance(pc.colors_, pc2.colors_), colors_max_error);
    }

    // test writing if we only have normals or colors that we are comparing
    if (!(int(args.compare) & int(Compare::NORMALS))) {
        pc2.normals_.clear();
    }
    if (!(int(args.compare) & int(Compare::COLORS))) {
        pc2.colors_.clear();
    }

    // Loaded data when saved should be identical when reloaded
    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc2,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc3;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc3,
                               {"auto", false, false, true}));
    EXPECT_EQ(MaxDistance(pc3.points_, pc2.points_), 0);
    if (int(args.compare) & int(Compare::NORMALS)) {
        SCOPED_TRACE("Normals");
        EXPECT_EQ(MaxDistance(pc3.normals_, pc2.normals_), 0);
    }
    if (int(args.compare) & int(Compare::COLORS)) {
        SCOPED_TRACE("Colors");
        EXPECT_EQ(MaxDistance(pc3.colors_, pc2.colors_), 0);
    }
}

// Most formats store color as uint8_t (0-255), while we store it as double
// [0.,1.] c_double=c_uint8/255.; however to go back, if we use
// c_uint8=c_double*255. then floating point error can produce a slightly lower
// number which will end up being 1 lower then what it should be.  These tests
// check that all formats properly round (instead of floor) color if they
// convert to c_uint8.

// save, load, save, load
TEST_P(ReadWritePC, ColorReload) {
    ReadWritePCArgs args = GetParam();
    // skip formats that do not support color
    if (!(int(args.compare) & int(Compare::COLORS))) {
        return;
    }

    geometry::PointCloud pc_start;
    Eigen::Vector3d one(1, 1, 1);
    // we are working with 0-255, we should always be within 0.5/255.
    for (int i = 0; i < 256; ++i) {
        pc_start.points_.push_back(one * 0.);
        pc_start.colors_.push_back(one * ((i) / 255.0));
    }

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();

    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc_start,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc_load;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc_load,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc_load.colors_) * 255., .5);

    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc_load,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc.colors_) * 255, .5);
}

// writing as another format (.xyzrgb) then loading and writing as specified
// format
TEST_P(ReadWritePC, ColorConvertLoad) {
    ReadWritePCArgs args = GetParam();
    // skip formats that do not support color
    if (!(int(args.compare) & int(Compare::COLORS))) {
        return;
    }

    geometry::PointCloud pc_start;
    Eigen::Vector3d one(1, 1, 1);
    // we are working with 0-255, we should always be within 0.5/255.
    for (int i = 0; i < 256; ++i) {
        pc_start.points_.push_back(one * 0.);
        pc_start.colors_.push_back(one * ((i) / 255.0));
    }

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();
    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc_start,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc_load;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc_load,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc_load.colors_) * 255., .5);

    EXPECT_TRUE(WritePointCloud(tmp_path + "/test0.xyzrgb", pc_load,
                                {true, false, true}));
    geometry::PointCloud pc2;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/test0.xyzrgb", pc2,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc2.colors_) * 255., .5);

    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc2,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc3;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc3,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc3.colors_) * 255., .5);
}

// avg on grayscale, then writing and loading
TEST_P(ReadWritePC, ColorGrayAvg) {
    ReadWritePCArgs args = GetParam();
    // skip formats that do not support color
    if (!(int(args.compare) & int(Compare::COLORS))) {
        return;
    }

    geometry::PointCloud pc_start;
    Eigen::Vector3d one(1, 1, 1);
    // we are working with 0-255, we should always be within 0.5/255.
    for (int i = 0; i < 256; ++i) {
        pc_start.points_.push_back(one * 0.);
        pc_start.colors_.push_back(one * ((i) / 255.0));
    }

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();
    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc_start,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc_load;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc_load,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc_load.colors_) * 255., .5);

    geometry::PointCloud pc_avg_col = pc_load;
    for (auto &c : pc_avg_col.colors_) {
        double avg = (c[0] + c[1] + c[2]) / 3.;
        c[0] = c[1] = c[2] = avg;
    }
    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc_avg_col,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc.colors_) * 255, .5);
}

// grayscale luma on grayscale, then writing and loading
TEST_P(ReadWritePC, ColorGrayscaleLuma) {
    ReadWritePCArgs args = GetParam();
    // skip formats that do not support color
    if (!(int(args.compare) & int(Compare::COLORS))) {
        return;
    }

    geometry::PointCloud pc_start;
    Eigen::Vector3d one(1, 1, 1);
    // we are working with 0-255, we should always be within 0.5/255.
    for (int i = 0; i < 256; ++i) {
        pc_start.points_.push_back(one * 0.);
        pc_start.colors_.push_back(one * ((i) / 255.0));
    }

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();
    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc_start,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc_load;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc_load,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc_load.colors_) * 255., .5);

    geometry::PointCloud pc_avg_col = pc_load;
    for (auto &c : pc_avg_col.colors_) {
        double gray = .2126 * c[0] + .7152 * c[1] + .0722 * c[2];
        c[0] = c[1] = c[2] = gray;
    }
    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc_avg_col,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc,
                               {"auto", false, false, true}));
    EXPECT_LT(MaxDistance(pc_start.colors_, pc.colors_) * 255, .5);
}

// crop instead of wraparound, color -.5 ends up <=0, color 1.5 ends up >=1
TEST_P(ReadWritePC, ColorCrop) {
    ReadWritePCArgs args = GetParam();
    // skip formats that do not support color
    if (!(int(args.compare) & int(Compare::COLORS))) {
        return;
    }

    geometry::PointCloud pc_start;
    Eigen::Vector3d one(1, 1, 1);
    pc_start.points_.push_back(one * 0.);
    pc_start.colors_.push_back(one * (-.5));
    pc_start.points_.push_back(one * 0.);
    pc_start.colors_.push_back(one * (1.5));

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();
    EXPECT_TRUE(WritePointCloud(
            tmp_path + "/" + args.filename, pc_start,
            {bool(args.write_ascii), bool(args.compressed), true}));
    geometry::PointCloud pc_load;
    EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc_load,
                               {"auto", false, false, true}));
    EXPECT_LT(pc_load.colors_[0](0), .001);
    EXPECT_GT(pc_load.colors_[1](0), .999);
}

// +-0.2/255. should round and be within 0.5 (note it maybe off by
// sqrt(3*.2^2)=.346)
TEST_P(ReadWritePC, ColorRounding) {
    ReadWritePCArgs args = GetParam();
    // skip formats that do not support color
    if (!(int(args.compare) & int(Compare::COLORS))) {
        return;
    }

    {
        geometry::PointCloud pc_start;
        Eigen::Vector3d one(1, 1, 1);
        // we are working with 0-255, we should always be within 0.5/255.
        for (int i = 0; i < 256; ++i) {
            pc_start.points_.push_back(one * 0.);
            pc_start.colors_.push_back(one * ((i - 0.2) / 255.0));
        }

        const std::string tmp_path =
                utility::filesystem::GetTempDirectoryPath();
        EXPECT_TRUE(WritePointCloud(
                tmp_path + "/" + args.filename, pc_start,
                {bool(args.write_ascii), bool(args.compressed), true}));
        geometry::PointCloud pc_load;
        EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc_load,
                                   {"auto", false, false, true}));
        EXPECT_LT(MaxDistance(pc_start.colors_, pc_load.colors_) * 255., .5);
    }
    {
        geometry::PointCloud pc_start;
        Eigen::Vector3d one(1, 1, 1);
        // we are working with 0-255, we should always be within 0.5/255.
        for (int i = 0; i < 256; ++i) {
            pc_start.points_.push_back(one * 0.);
            pc_start.colors_.push_back(one * ((i + 0.2) / 255.0));
        }

        const std::string tmp_path =
                utility::filesystem::GetTempDirectoryPath();
        EXPECT_TRUE(WritePointCloud(
                tmp_path + "/" + args.filename, pc_start,
                {bool(args.write_ascii), bool(args.compressed), true}));
        geometry::PointCloud pc_load;
        EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc_load,
                                   {"auto", false, false, true}));
        EXPECT_LT(MaxDistance(pc_start.colors_, pc_load.colors_) * 255., .5);
    }
}

TEST_P(ReadWritePC, UpdateProgressCallback) {
    ReadWritePCArgs args = GetParam();
    geometry::PointCloud pc;
    RandPC(pc, 32 * 1024);

    double last_percent;
    int num_calls;
    auto Clear = [&]() { last_percent = num_calls = 0; };
    auto Update = [&](double percent) {
        last_percent = percent;
        ++num_calls;
        return true;
    };

    {
        WritePointCloudOption p(bool(args.write_ascii), bool(args.compressed));
        p.update_progress = Update;
        Clear();
        const std::string tmp_path =
                utility::filesystem::GetTempDirectoryPath();
        EXPECT_TRUE(WritePointCloud(tmp_path + "/" + args.filename, pc, p));
        EXPECT_EQ(last_percent, 100.);
        EXPECT_GT(num_calls, 10);
    }
    {
        ReadPointCloudOption p(Update);
        Clear();
        const std::string tmp_path =
                utility::filesystem::GetTempDirectoryPath();
        EXPECT_TRUE(ReadPointCloud(tmp_path + "/" + args.filename, pc, p));
        EXPECT_EQ(last_percent, 100.);
        EXPECT_GT(num_calls, 10);
    }
}

TEST(PointCloudIO, DISABLED_CreatePointCloudFromFile) { NotImplemented(); }

}  // namespace tests
}  // namespace open3d
