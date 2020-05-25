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

#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/Geometry/PointCloud.h"
#include "TestUtility/UnitTest.h"

namespace open3d {
namespace unit_test {

using open3d::io::ReadPointCloud;
using open3d::io::WritePointCloud;

namespace {

template <class T>
double AverageDistance(const std::vector<T> &a, const std::vector<T> &b) {
    // Note: cannot use ASSERT_EQ because we return non-void
    EXPECT_EQ(a.size(), b.size());
    double total = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        total += (a[i] - b[i]).norm();
    }
    return total / a.size();
}

void RandPC(geometry::PointCloud &pc) {
    int size = 100;

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
        {"test.pcd", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 0
        {"test.pcd", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 1
        {"test.pcd", IsAscii::BINARY, Compressed::COMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 2
        {"test.pcd", IsAscii::ASCII, Compressed::COMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 3
        {"test.ply", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 4
        {"test.ply", IsAscii::ASCII, Compressed::UNCOMPRESSED,
         Compare::NORMALS_AND_COLORS},  // 5
        {"test.pts", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::COLORS},  // 6
        {"test.xyz", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NONE},  // 7
        {"test.xyzn", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::NORMALS},  // 8
        {"test.xyzrgb", IsAscii::BINARY, Compressed::UNCOMPRESSED,
         Compare::COLORS},  // 9
});

class ReadWritePC : public testing::TestWithParam<ReadWritePCArgs> {};
INSTANTIATE_TEST_SUITE_P(ReadWritePC, ReadWritePC, testing::ValuesIn(pcArgs));

TEST_P(ReadWritePC, Basic) {
    ReadWritePCArgs args = GetParam();
    geometry::PointCloud pc;
    RandPC(pc);

    // we loose some precision when saving generated data
    EXPECT_TRUE(WritePointCloud(args.filename, pc, bool(args.write_ascii),
                                bool(args.compressed), true));
    geometry::PointCloud pc2;
    EXPECT_TRUE(ReadPointCloud(args.filename, pc2, "auto", false, false, true));
    const double pointsMaxError =
            1e-3;  //.ply ascii has the highest error, others <1e-4
    EXPECT_LT(AverageDistance(pc.points_, pc2.points_), pointsMaxError);
    if (int(args.compare) & int(Compare::NORMALS)) {
        SCOPED_TRACE("Normals");
        const double normalsMaxError =
                1e-6;  //.ply ascii has the highest error, others <1e-7
        EXPECT_LT(AverageDistance(pc.normals_, pc2.normals_), normalsMaxError);
    }
    if (int(args.compare) & int(Compare::COLORS)) {
        SCOPED_TRACE("Colors");
        const double colorsMaxError =
                1e-2;  // colors are saved as uint8_t[3] in a lot of formats
        EXPECT_LT(AverageDistance(pc.colors_, pc2.colors_), colorsMaxError);
    }

    // Loaded data when saved should be identical when reloaded
    EXPECT_TRUE(WritePointCloud(args.filename, pc2, bool(args.write_ascii),
                                bool(args.compressed), true));
    geometry::PointCloud pc3;
    EXPECT_TRUE(ReadPointCloud(args.filename, pc3, "auto", false, false, true));
    EXPECT_EQ(AverageDistance(pc3.points_, pc2.points_), 0);
    if (int(args.compare) & int(Compare::NORMALS)) {
        SCOPED_TRACE("Normals");
        EXPECT_EQ(AverageDistance(pc3.normals_, pc2.normals_), 0);
    }
    if (int(args.compare) & int(Compare::COLORS)) {
        SCOPED_TRACE("Colors");
        EXPECT_EQ(AverageDistance(pc3.colors_, pc2.colors_), 0);
    }
}

TEST(PointCloudIO, DISABLED_CreatePointCloudFromFile) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_ReadPointCloud) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_WritePointCloud) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_ReadPointCloudFromXYZ) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_WritePointCloudToXYZ) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_ReadPointCloudFromXYZN) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_WritePointCloudToXYZN) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_ReadPointCloudFromXYZRGB) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_WritePointCloudToXYZRGB) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_ReadPointCloudFromPLY) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_WritePointCloudToPLY) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_ReadPointCloudFromPCD) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_WritePointCloudToPCD) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_ReadPointCloudFromPTS) { NotImplemented(); }

TEST(PointCloudIO, DISABLED_WritePointCloudToPTS) { NotImplemented(); }

}  // namespace unit_test
}  // namespace open3d
