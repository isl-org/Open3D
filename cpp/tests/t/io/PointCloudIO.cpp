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

#include "open3d/t/io/PointCloudIO.h"

#include <gtest/gtest.h>

#include "core/CoreTest.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"
#include "open3d/t/geometry/PointCloud.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

namespace {

class PointCloudIOPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(PointCloudIO,
                         PointCloudIOPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

struct TensorCtorData {
    std::vector<double> values;
    core::SizeVector size;
};

enum class IsAscii : bool { BINARY = false, ASCII = true };
enum class Compressed : bool { UNCOMPRESSED = false, COMPRESSED = true };
struct ReadWritePCArgs {
    std::string filename;
    IsAscii write_ascii;
    Compressed compressed;
    std::unordered_map<std::string, double> attributes_rel_tols;
};

}  // namespace

const std::unordered_map<std::string, TensorCtorData> pc_data_1{
        {"points", {{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, {5, 3}}},
        {"intensities", {{0, 0.5, 0.5, 0.5, 1}, {5, 1}}}};

// Bad data.
const std::unordered_map<std::string, TensorCtorData> pc_data_bad{
        {"points", {{0, 0, 0, 1, 0, 0}, {2, 3}}},
        {"intensities", {{0}, {1, 1}}},
};

const std::vector<ReadWritePCArgs> pcArgs({
        {"test.xyzi",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"points", 1e-5}, {"intensities", 1e-5}}},  // 0
        {"test.ply",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"points", 1e-5}, {"intensities", 1e-5}}},  // 1
});

class ReadWriteTPC : public testing::TestWithParam<ReadWritePCArgs> {};
INSTANTIATE_TEST_SUITE_P(ReadWritePC, ReadWriteTPC, testing::ValuesIn(pcArgs));

TEST_P(ReadWriteTPC, Basic) {
    ReadWritePCArgs args = GetParam();
    core::Device device("CPU", 0);
    core::Dtype dtype = core::Dtype::Float64;
    t::geometry::PointCloud pc1(device);

    for (const auto &attr_tensor : pc_data_1) {
        const auto &attr = attr_tensor.first;
        const auto &tensor = attr_tensor.second;
        pc1.SetPointAttr(
                attr, core::Tensor(tensor.values, tensor.size, dtype, device));
    }

    // we loose some precision when saving generated data
    // test writing if we have point, normal, and colors in pc
    EXPECT_TRUE(t::io::WritePointCloud(
            args.filename, pc1,
            {bool(args.write_ascii), bool(args.compressed), true}));
    t::geometry::PointCloud pc2(device);
    EXPECT_TRUE(t::io::ReadPointCloud(args.filename, pc2,
                                      {"auto", false, false, true}));

    for (const auto &attribute_rel_tol : args.attributes_rel_tols) {
        const std::string &attribute = attribute_rel_tol.first;
        const double rel_tol = attribute_rel_tol.second;
        SCOPED_TRACE(attribute);
        EXPECT_TRUE(pc1.GetPointAttr(attribute).AllClose(
                pc2.GetPointAttr(attribute), rel_tol));
    }

    // Loaded data when saved should be identical when reloaded
    EXPECT_TRUE(t::io::WritePointCloud(
            args.filename, pc2,
            {bool(args.write_ascii), bool(args.compressed), true}));
    t::geometry::PointCloud pc3(device);
    EXPECT_TRUE(t::io::ReadPointCloud(args.filename, pc3,
                                      {"auto", false, false, true}));
    for (const auto &attribute_rel_tol : args.attributes_rel_tols) {
        const std::string &attribute = attribute_rel_tol.first;
        SCOPED_TRACE(attribute);
        EXPECT_TRUE(pc3.GetPointAttr(attribute).AllClose(
                pc2.GetPointAttr(attribute), 0, 0));
    }
}

TEST_P(ReadWriteTPC, WriteBadData) {
    ReadWritePCArgs args = GetParam();
    core::Device device("CPU", 0);
    core::Dtype dtype = core::Dtype::Float64;
    t::geometry::PointCloud pc1(device);

    for (const auto &attr_tensor : pc_data_bad) {
        const auto &attr = attr_tensor.first;
        const auto &tensor = attr_tensor.second;
        pc1.SetPointAttr(
                attr, core::Tensor(tensor.values, tensor.size, dtype, device));
    }

    EXPECT_FALSE(t::io::WritePointCloud(
            args.filename, pc1,
            {bool(args.write_ascii), bool(args.compressed), true}));
}

// Reading binary_little_endian with colors and normals.
TEST(TPointCloudIO, ReadPointCloudFromPLY1) {
    t::geometry::PointCloud pcd;

    t::io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/fragment.ply", pcd,
                          {"auto", false, false, true});
    EXPECT_EQ(pcd.GetPoints().GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointNormals().GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointColors().GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointAttr("curvature").GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointColors().GetDtype(), core::Dtype::UInt8);
    EXPECT_FALSE(pcd.HasPointAttr("x"));
}

// Reading ascii.
TEST(TPointCloudIO, ReadPointCloudFromPLY2) {
    t::geometry::PointCloud pcd;

    t::io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/test_sample_ascii.ply",
                          pcd, {"auto", false, false, true});
    EXPECT_EQ(pcd.GetPoints().GetLength(), 7);
}

// Skip unsupported datatype.
TEST(TPointCloudIO, ReadPointCloudFromPLY3) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(
            std::string(TEST_DATA_DIR) + "/test_sample_wrong_format.ply", pcd,
            {"auto", false, false, true});
    EXPECT_FALSE(pcd.HasPointAttr("intensity"));
}

// Custom attributes check.
TEST(TPointCloudIO, ReadPointCloudFromPLY4) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(
            std::string(TEST_DATA_DIR) + "/test_sample_custom.ply", pcd,
            {"auto", false, false, true});
    EXPECT_EQ(pcd.GetPoints().GetLength(), 7);
    EXPECT_EQ(pcd.GetPointAttr("intensity").GetLength(), 7);
}

// Read write empty point cloud.
TEST(TPointCloudIO, ReadWriteEmptyPTS) {
    t::geometry::PointCloud pcd, pcd_read;
    std::string file_name = std::string(TEST_DATA_DIR) + "/test_empty.pts";
    EXPECT_TRUE(pcd.IsEmpty());
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd_read.IsEmpty());
    std::remove(file_name.c_str());
}

// Read write pts with colors and intensities.
TEST(TPointCloudIO, ReadWritePTS) {
    t::geometry::PointCloud pcd, pcd_read, pcd_i, pcd_color;
    EXPECT_TRUE(t::io::ReadPointCloud(
            std::string(TEST_DATA_DIR) +
                    "/open3d_downloads/tests/point_cloud_sample1.pts",
            pcd, {"auto", false, false, true}));
    EXPECT_EQ(pcd.GetPoints().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointColors().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointAttr("intensities").GetLength(), 10);
    EXPECT_EQ(pcd.GetPointColors().GetDtype(), core::Dtype::UInt8);

    // Write pointcloud and match it after read.
    std::string file_name = std::string(TEST_DATA_DIR) + "/test_read.pts";
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd.GetPoints().AllClose(pcd_read.GetPoints()));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(pcd_read.GetPointColors()));
    EXPECT_TRUE(pcd.GetPointAttr("intensities")
                        .AllClose(pcd_read.GetPointAttr("intensities")));
    std::remove(file_name.c_str());

    // Write pointcloud with only colors and match it after read.
    pcd_read.Clear();
    pcd_color.SetPoints(pcd.GetPoints());
    pcd_color.SetPointColors(pcd.GetPointColors());
    file_name = std::string(TEST_DATA_DIR) + "/test_color.pts";
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd_color));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd_color.GetPoints().AllClose(pcd_read.GetPoints()));
    EXPECT_TRUE(pcd_color.GetPointColors().AllClose(pcd_read.GetPointColors()));
    EXPECT_FALSE(pcd_read.HasPointAttr("intensities"));
    std::remove(file_name.c_str());

    // Write pointcloud with only intensities and match it after read.
    pcd_read.Clear();
    pcd_i.SetPoints(pcd.GetPoints());
    pcd_i.SetPointAttr("intensities", pcd.GetPointAttr("intensities"));
    file_name = std::string(TEST_DATA_DIR) + "/test_intensities.pts";
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd_i));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd_i.GetPoints().AllClose(pcd_read.GetPoints()));
    EXPECT_TRUE(pcd_i.GetPointAttr("intensities")
                        .AllClose(pcd_read.GetPointAttr("intensities")));
    EXPECT_FALSE(pcd_read.HasPointColors());
    std::remove(file_name.c_str());
}

// Reading pts with intensities.
TEST(TPointCloudIO, ReadPointCloudFromPTS1) {
    t::geometry::PointCloud pcd;
    EXPECT_TRUE(t::io::ReadPointCloud(
            std::string(TEST_DATA_DIR) +
                    "/open3d_downloads/tests/point_cloud_sample2.pts",
            pcd, {"auto", false, false, true}));
    EXPECT_EQ(pcd.GetPoints().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointAttr("intensities").GetLength(), 10);
}

// Reading bunny pts.
TEST(TPointCloudIO, ReadPointCloudFromPTS2) {
    t::geometry::PointCloud pcd;
    EXPECT_TRUE(t::io::ReadPointCloud(
            std::string(TEST_DATA_DIR) +
                    "/open3d_downloads/tests/bunnyData.pts",
            pcd, {"auto", false, false, true}));
    EXPECT_EQ(pcd.GetPoints().GetLength(), 30571);
}

// Check PTS color float to uint8 conversion.
TEST(TPointCloudIO, WritePTSColorConversion1) {
    t::geometry::PointCloud pcd, pcd_read;
    std::string file_name =
            std::string(TEST_DATA_DIR) + "/test_color_conversion.pts";
    pcd.SetPoints(core::Tensor::Init<double>({{1, 2, 3}, {4, 5, 6}}));
    pcd.SetPointColors(
            core::Tensor::Init<float>({{-1, 0.25, 0.3}, {0, 4, 0.1}}));
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_EQ(pcd_read.GetPointColors().ToFlatVector<uint8_t>(),
              std::vector<uint8_t>({0, 64, 77, 0, 255, 26}));
    std::remove(file_name.c_str());
}

// Check PTS color boolean to uint8 conversion.
TEST(TPointCloudIO, WritePTSColorConversion2) {
    t::geometry::PointCloud pcd, pcd_read;
    std::string file_name =
            std::string(TEST_DATA_DIR) + "/test_color_conversion.pts";
    pcd.SetPoints(core::Tensor::Init<double>({{1, 2, 3}, {4, 5, 6}}));
    pcd.SetPointColors(core::Tensor::Init<bool>({{1, 0, 0}, {1, 0, 1}}));
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_EQ(pcd_read.GetPointColors().ToFlatVector<uint8_t>(),
              std::vector<uint8_t>({255, 0, 0, 255, 0, 255}));
    std::remove(file_name.c_str());
}

TEST_P(PointCloudIOPermuteDevices, WriteDeviceTestPLY) {
    core::Device device = GetParam();
    std::string filename = std::string(TEST_DATA_DIR) + "/test_write.ply";
    core::Tensor points =
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device);
    t::geometry::PointCloud pcd(points);
    EXPECT_TRUE(t::io::WritePointCloud(filename, pcd));
    std::remove(filename.c_str());
}

}  // namespace tests
}  // namespace open3d
