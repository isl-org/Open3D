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

#include "open3d/t/io/PointCloudIO.h"

#include <gtest/gtest.h>

#include "core/CoreTest.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/t/geometry/PointCloud.h"
#include "tests/Tests.h"

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
        {"positions", {{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, {5, 3}}},
        {"intensities", {{0, 0.5, 0.5, 0.5, 1}, {5, 1}}}};

// Bad data.
const std::unordered_map<std::string, TensorCtorData> pc_data_bad{
        {"positions", {{0, 0, 0, 1, 0, 0}, {2, 3}}},
        {"intensities", {{0}, {1, 1}}},
};

const std::vector<ReadWritePCArgs> pcArgs({
        {"test.xyzi",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"positions", 1e-5}, {"intensities", 1e-5}}},  // 0
        {"test.ply",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"positions", 1e-5}, {"intensities", 1e-5}}},  // 1
});

class ReadWriteTPC : public testing::TestWithParam<ReadWritePCArgs> {};
INSTANTIATE_TEST_SUITE_P(ReadWritePC, ReadWriteTPC, testing::ValuesIn(pcArgs));

TEST_P(ReadWriteTPC, Basic) {
    ReadWritePCArgs args = GetParam();
    core::Device device("CPU:0");
    core::Dtype dtype = core::Float64;
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
    core::Device device("CPU:0");
    core::Dtype dtype = core::Float64;
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

    data::PLYPointCloud pointcloud_ply;
    t::io::ReadPointCloud(pointcloud_ply.GetPath(), pcd,
                          {"auto", false, false, true});
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointNormals().GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointColors().GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointAttr("curvature").GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointColors().GetDtype(), core::UInt8);
    EXPECT_FALSE(pcd.HasPointAttr("x"));
}

// Reading ascii.
TEST(TPointCloudIO, ReadPointCloudFromPLY2) {
    t::geometry::PointCloud pcd;

    t::io::ReadPointCloud(utility::GetDataPathCommon("test_sample_ascii.ply"),
                          pcd, {"auto", false, false, true});
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 7);
}

// Skip unsupported datatype.
TEST(TPointCloudIO, ReadPointCloudFromPLY3) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(
            utility::GetDataPathCommon("test_sample_wrong_format.ply"), pcd,
            {"auto", false, false, true});
    EXPECT_FALSE(pcd.HasPointAttr("intensity"));
}

// Custom attributes check.
TEST(TPointCloudIO, ReadPointCloudFromPLY4) {
    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(utility::GetDataPathCommon("test_sample_custom.ply"),
                          pcd, {"auto", false, false, true});
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 7);
    EXPECT_EQ(pcd.GetPointAttr("intensity").GetLength(), 7);
}

// Read write empty point cloud.
TEST(TPointCloudIO, ReadWriteEmptyPTS) {
    t::geometry::PointCloud pcd, pcd_read;
    std::string file_name = utility::GetDataPathCommon("test_empty.pts");
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
            utility::GetDataPathDownload("tests/point_cloud_sample1.pts"), pcd,
            {"auto", false, false, true}));
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointColors().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointAttr("intensities").GetLength(), 10);
    EXPECT_EQ(pcd.GetPointColors().GetDtype(), core::UInt8);
    EXPECT_TRUE(pcd.GetPointPositions()[0].AllClose(
            core::Tensor::Init<double>({4.24644, -6.42662, -50.2146})));
    EXPECT_TRUE(pcd.GetPointColors()[0].AllClose(
            core::Tensor::Init<uint8_t>({66, 50, 83})));
    EXPECT_TRUE(pcd.GetPointAttr("intensities")[0].AllClose(
            core::Tensor::Init<double>({10})));

    // Write pointcloud and match it after read.
    std::string file_name = utility::GetDataPathCommon("test_read.pts");
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd.GetPointPositions().AllClose(pcd_read.GetPointPositions()));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(pcd_read.GetPointColors()));
    EXPECT_TRUE(pcd.GetPointAttr("intensities")
                        .AllClose(pcd_read.GetPointAttr("intensities")));
    std::remove(file_name.c_str());

    // Write pointcloud with only colors and match it after read.
    pcd_read.Clear();
    pcd_color.SetPointPositions(pcd.GetPointPositions());
    pcd_color.SetPointColors(pcd.GetPointColors());
    file_name = utility::GetDataPathCommon("test_color.pts");
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd_color));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd_color.GetPointPositions().AllClose(
            pcd_read.GetPointPositions()));
    EXPECT_TRUE(pcd_color.GetPointColors().AllClose(pcd_read.GetPointColors()));
    EXPECT_FALSE(pcd_read.HasPointAttr("intensities"));
    std::remove(file_name.c_str());

    // Write pointcloud with only intensities and match it after read.
    pcd_read.Clear();
    pcd_i.SetPointPositions(pcd.GetPointPositions());
    pcd_i.SetPointAttr("intensities", pcd.GetPointAttr("intensities"));
    file_name = utility::GetDataPathCommon("test_intensities.pts");
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd_i));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(
            pcd_i.GetPointPositions().AllClose(pcd_read.GetPointPositions()));
    EXPECT_TRUE(pcd_i.GetPointAttr("intensities")
                        .AllClose(pcd_read.GetPointAttr("intensities")));
    EXPECT_FALSE(pcd_read.HasPointColors());
    std::remove(file_name.c_str());
}

// Reading pts with intensities.
TEST(TPointCloudIO, ReadPointCloudFromPTS1) {
    t::geometry::PointCloud pcd;
    EXPECT_TRUE(t::io::ReadPointCloud(
            utility::GetDataPathDownload("tests/point_cloud_sample2.pts"), pcd,
            {"auto", false, false, true}));
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointAttr("intensities").GetLength(), 10);
}

// Reading bunny pts.
TEST(TPointCloudIO, ReadPointCloudFromPTS2) {
    t::geometry::PointCloud pcd;
    EXPECT_TRUE(t::io::ReadPointCloud(
            utility::GetDataPathDownload("tests/bunnyData.pts"), pcd,
            {"auto", false, false, true}));
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 30571);
}

// Check PTS color float to uint8 conversion.
TEST(TPointCloudIO, WritePTSColorConversion1) {
    t::geometry::PointCloud pcd, pcd_read;
    std::string file_name =
            utility::GetDataPathCommon("test_color_conversion.pts");
    pcd.SetPointPositions(core::Tensor::Init<double>({{1, 2, 3}, {4, 5, 6}}));
    pcd.SetPointColors(
            core::Tensor::Init<float>({{-1, 0.25, 0.4}, {0, 4, 0.1}}));
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_EQ(pcd_read.GetPointColors().ToFlatVector<uint8_t>(),
              std::vector<uint8_t>({0, 64, 102, 0, 255, 26}));
    std::remove(file_name.c_str());
}

// Check PTS color boolean to uint8 conversion.
TEST(TPointCloudIO, WritePTSColorConversion2) {
    t::geometry::PointCloud pcd, pcd_read;
    std::string file_name =
            utility::GetDataPathCommon("test_color_conversion.pts");
    pcd.SetPointPositions(core::Tensor::Init<double>({{1, 2, 3}, {4, 5, 6}}));
    pcd.SetPointColors(core::Tensor::Init<bool>({{1, 0, 0}, {1, 0, 1}}));
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_EQ(pcd_read.GetPointColors().ToFlatVector<uint8_t>(),
              std::vector<uint8_t>({255, 0, 0, 255, 0, 255}));
    std::remove(file_name.c_str());
}

TEST(TPointCloudIO, ReadWritePointCloudAsNPZ) {
    // Read PointCloud from PLY file.
    t::geometry::PointCloud pcd_ply;
    data::PLYPointCloud pointcloud_ply;
    t::io::ReadPointCloud(pointcloud_ply.GetPath(), pcd_ply,
                          {"auto", false, false, true});

    core::Tensor custom_attr = core::Tensor::Ones(
            pcd_ply.GetPointPositions().GetShape(), core::Float32);
    pcd_ply.SetPointAttr("custom_attr", custom_attr);

    std::string filename =
            utility::GetDataPathCommon("test_npz_pointcloud.npz");
    EXPECT_TRUE(t::io::WritePointCloud(filename, pcd_ply));

    // Read from the saved pointcloud.
    t::geometry::PointCloud pcd_npz;
    t::io::ReadPointCloud(filename, pcd_npz, {"auto", false, false, true});

    for (auto &kv : pcd_ply.GetPointAttr()) {
        EXPECT_TRUE(kv.second.AllClose(pcd_npz.GetPointAttr(kv.first)));
    }

    std::remove(filename.c_str());
}

TEST_P(PointCloudIOPermuteDevices, WriteDeviceTestPLY) {
    core::Device device = GetParam();
    std::string filename = utility::GetDataPathCommon("test_write.ply");
    core::Tensor points = core::Tensor::Ones({10, 3}, core::Float32, device);
    t::geometry::PointCloud pcd(points);
    EXPECT_TRUE(t::io::WritePointCloud(filename, pcd));
    std::remove(filename.c_str());
}

TEST(TPointCloudIO, ReadWritePointCloudAsPCD) {
    // Read PointCloud from PLY file.
    t::geometry::PointCloud input_pcd;
    // Using PLY Read to load the data.
    data::PLYPointCloud pointcloud_ply;
    t::io::ReadPointCloud(pointcloud_ply.GetPath(), input_pcd,
                          {"auto", false, false, false});

    // Adding custom attributes of different dtypes.
    core::Tensor custom_attr_uint8 = core::Tensor::Ones(
            {input_pcd.GetPointPositions().GetLength(), 1}, core::UInt8);
    input_pcd.SetPointAttr("custom_attr_uint8", custom_attr_uint8);

    core::Tensor custom_attr_uint32 = core::Tensor::Ones(
            {input_pcd.GetPointPositions().GetLength(), 1}, core::UInt32);
    input_pcd.SetPointAttr("custom_attr_uint32", custom_attr_uint32);

    core::Tensor custom_attr_int32 = core::Tensor::Ones(
            {input_pcd.GetPointPositions().GetLength(), 1}, core::Int32);
    input_pcd.SetPointAttr("custom_attr_int32", custom_attr_int32);

    core::Tensor custom_attr_int64 = core::Tensor::Ones(
            {input_pcd.GetPointPositions().GetLength(), 1}, core::Int64);
    input_pcd.SetPointAttr("custom_attr_int64", custom_attr_int64);

    core::Tensor custom_attr_float = core::Tensor::Ones(
            {input_pcd.GetPointPositions().GetLength(), 1}, core::Float32);
    input_pcd.SetPointAttr("custom_attr_float", custom_attr_float);

    core::Tensor custom_attr_double = core::Tensor::Ones(
            {input_pcd.GetPointPositions().GetLength(), 1}, core::Float64);
    input_pcd.SetPointAttr("custom_attr_double", custom_attr_double);

    // PCD IO for ASCII format.
    std::string filename_ascii =
            utility::GetDataPathCommon("test_pcd_pointcloud_ascii.pcd");

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_ascii, input_pcd,
            open3d::io::WritePointCloudOption(
                    /*ascii*/ true, /*compressed*/ false, false, {})));

    t::geometry::PointCloud ascii_pcd;
    t::io::ReadPointCloud(filename_ascii, ascii_pcd);

    for (auto &kv : input_pcd.GetPointAttr()) {
        EXPECT_TRUE(kv.second.AllClose(ascii_pcd.GetPointAttr(kv.first), 1e-3,
                                       1e-4));
    }
    std::remove(filename_ascii.c_str());

    // PCD IO for Binary format.
    std::string filename_binary =
            utility::GetDataPathCommon("test_pcd_pointcloud_binary.pcd");

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_binary, input_pcd,
            open3d::io::WritePointCloudOption(
                    /*ascii*/ false, /*compressed*/ false, false, {})));

    t::geometry::PointCloud binary_pcd;
    t::io::ReadPointCloud(filename_binary, binary_pcd);

    for (auto &kv : input_pcd.GetPointAttr()) {
        EXPECT_TRUE(kv.second.AllClose(binary_pcd.GetPointAttr(kv.first), 1e-3,
                                       1e-4));
    }
    std::remove(filename_binary.c_str());

    // PCD IO for Binary Compressed format.
    std::string filename_binary_compressed = utility::GetDataPathCommon(
            "test_pcd_pointcloud_binary_compressed.pcd");

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_binary_compressed, input_pcd,
            open3d::io::WritePointCloudOption(
                    /*ascii*/ false, /*compressed*/ true, false, {})));

    t::geometry::PointCloud binary_compressed_pcd;
    t::io::ReadPointCloud(filename_binary_compressed, binary_compressed_pcd);

    for (auto &kv : input_pcd.GetPointAttr()) {
        EXPECT_TRUE(kv.second.AllClose(
                binary_compressed_pcd.GetPointAttr(kv.first), 1e-3, 1e-4));
    }
    std::remove(filename_binary_compressed.c_str());

    // Colors data type will be converted to UInt8 during Write / Read.
    // Only Float32, Float64, UInt8, UInt16, UInt32 colors data types are
    // suppoted for conversion.

    // PointCloud with Float32 type white color
    core::Tensor color_float32 = core::Tensor::Ones(
            {input_pcd.GetPointPositions().GetLength(), 3}, core::Float32);
    input_pcd.SetPointColors(color_float32);

    std::string filename_ascii_f32 = utility::GetDataPathCommon(
            "test_pcd_pointcloud_binary_f32_colors.pcd");

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_ascii_f32, input_pcd,
            open3d::io::WritePointCloudOption(
                    /*ascii*/ true, /*compressed*/ false, false, {})));

    t::geometry::PointCloud ascii_f32_pcd;
    t::io::ReadPointCloud(filename_ascii_f32, ascii_f32_pcd);

    core::Tensor color_uint8 =
            color_float32.To(core::Dtype::UInt8)
                    .Mul(std::numeric_limits<uint8_t>::max());

    EXPECT_TRUE(ascii_f32_pcd.GetPointColors().AllClose(color_uint8));
    std::remove(filename_ascii_f32.c_str());

    // PointCloud with UInt32 type white color
    core::Tensor color_uint32 =
            color_float32.To(core::Dtype::UInt32)
                    .Mul(std::numeric_limits<uint32_t>::max());
    input_pcd.SetPointColors(color_uint32);

    std::string filename_ascii_uint32 = utility::GetDataPathCommon(
            "/test_pcd_pointcloud_binary_uint32_colors.pcd");

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_ascii_uint32, input_pcd,
            open3d::io::WritePointCloudOption(
                    /*ascii*/ true, /*compressed*/ false, false, {})));

    t::geometry::PointCloud ascii_uint32_pcd;
    t::io::ReadPointCloud(filename_ascii_uint32, ascii_uint32_pcd);

    EXPECT_TRUE(ascii_f32_pcd.GetPointColors().AllClose(color_uint8));
    std::remove(filename_ascii_uint32.c_str());
}

}  // namespace tests
}  // namespace open3d
