// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/PointCloudIO.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

#include "core/CoreTest.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

class PointCloudIOPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(PointCloudIO,
                         PointCloudIOPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

struct TensorCtorData {
    std::vector<float> values;
    core::SizeVector size;
};

enum class IsAscii : bool { BINARY = false, ASCII = true };
enum class Compressed : bool { UNCOMPRESSED = false, COMPRESSED = true };
struct ReadWritePCArgs {
    std::string filename;
    IsAscii write_ascii;
    Compressed compressed;
    std::unordered_map<std::string, float> attributes_rel_tols;
};

}  // namespace

const std::unordered_map<std::string, TensorCtorData> pc_data_1{
        {"positions", {{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, {5, 3}}},
        {"intensities", {{0, 0.5, 0.5, 0.5, 1}, {5, 1}}},
        {"colors",
         {{0.5, 0.3, 0.2, 0, 1, 0.5, 0.6, 0, 1, 1, 0.5, 0.7, 0.3, 0, 0.5},
          {5, 3}}},
        {"dopplers", {{1, 0.9, 0.8, 0.7, 0.6}, {5, 1}}},
};

// Bad data.
const std::unordered_map<std::string, TensorCtorData> pc_data_bad{
        {"positions", {{0, 0, 0, 1, 0, 0}, {2, 3}}},
        {"intensities", {{0}, {1, 1}}},
        {"dopplers", {{0}, {1, 1}}},
};

const std::vector<ReadWritePCArgs> pcArgs({
        {"test.xyzi",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"positions", 1e-5}, {"intensities", 1e-5}}},  // 0
        {"test.xyzrgb",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"positions", 1e-5}, {"colors", 1e-5}}},  // 1
        {"test.ply",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"positions", 1e-5}, {"intensities", 1e-5}, {"colors", 1e-5}}},  // 2
        {"test.xyzd",
         IsAscii::ASCII,
         Compressed::UNCOMPRESSED,
         {{"positions", 1e-5}, {"dopplers", 1e-5}}},  // 0
});

class ReadWriteTPC : public testing::TestWithParam<ReadWritePCArgs> {};
INSTANTIATE_TEST_SUITE_P(ReadWritePC, ReadWriteTPC, testing::ValuesIn(pcArgs));

TEST_P(ReadWriteTPC, Basic) {
    ReadWritePCArgs args = GetParam();
    core::Device device("CPU:0");
    core::Dtype dtype = core::Float32;
    t::geometry::PointCloud pc1(device);

    for (const auto &attr_tensor : pc_data_1) {
        const auto &attr = attr_tensor.first;
        const auto &tensor = attr_tensor.second;
        pc1.SetPointAttr(
                attr, core::Tensor(tensor.values, tensor.size, dtype, device));
    }

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();

    // we loose some precision when saving generated data
    // test writing if we have point, normal, and colors in pc
    EXPECT_TRUE(t::io::WritePointCloud(
            tmp_path + "/" + args.filename, pc1,
            {bool(args.write_ascii), bool(args.compressed), true}));

    t::geometry::PointCloud pc2(device);
    EXPECT_TRUE(t::io::ReadPointCloud(tmp_path + "/" + args.filename, pc2,
                                      {"auto", false, false, true}));

    for (const auto &attribute_rel_tol : args.attributes_rel_tols) {
        const std::string &attribute = attribute_rel_tol.first;
        const float rel_tol = attribute_rel_tol.second;
        SCOPED_TRACE(attribute);
        EXPECT_TRUE(pc1.GetPointAttr(attribute).AllClose(
                pc2.GetPointAttr(attribute), rel_tol));
    }

    // Loaded data when saved should be identical when reloaded
    EXPECT_TRUE(t::io::WritePointCloud(
            tmp_path + "/" + args.filename, pc2,
            {bool(args.write_ascii), bool(args.compressed), true}));
    t::geometry::PointCloud pc3(device);
    EXPECT_TRUE(t::io::ReadPointCloud(tmp_path + "/" + args.filename, pc3,
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
    core::Dtype dtype = core::Float32;
    t::geometry::PointCloud pc1(device);

    for (const auto &attr_tensor : pc_data_bad) {
        const auto &attr = attr_tensor.first;
        const auto &tensor = attr_tensor.second;
        pc1.SetPointAttr(
                attr, core::Tensor(tensor.values, tensor.size, dtype, device));
    }

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();

    EXPECT_FALSE(t::io::WritePointCloud(
            tmp_path + "/" + args.filename, pc1,
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

// Reading ascii and check for custom attributes.
TEST(TPointCloudIO, ReadPointCloudFromPLY2) {
    data::PLYPointCloud sample_ply_pointcloud;
    auto pcd_in =
            t::io::CreatePointCloudFromFile(sample_ply_pointcloud.GetPath());
    const std::string filename_out =
            utility::filesystem::GetTempDirectoryPath() + "/SampleASCII.ply";
    t::io::WritePointCloud(filename_out, *pcd_in,
                           {/*write_ascii =*/true, false});

    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(filename_out, pcd, {"auto", false, false, true});
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 196133);
    EXPECT_EQ(pcd.GetPointAttr("curvature").GetLength(), 196133);
}

// Skip unsupported datatype.
TEST(TPointCloudIO, ReadPointCloudFromPLY3) {
    std::string filename_out = utility::filesystem::GetTempDirectoryPath() +
                               "/test_sample_wrong_format.ply";
    char data[1000] =
            "ply \n"
            "format ascii 1.0 \n"
            "comment VCGLIB generated \n"
            "element vertex 2 \n"
            "property float x \n"
            "property float y \n"
            "property float z \n"
            "property char intensity \n"
            "property float nx \n"
            "property float ny \n"
            "property float nz \n"
            "end_header \n"
            "0 0 -1 100 0.003695 0 -4.16078 \n"
            "0.7236 -0.52572 -0.447215 127 -2.18747 -1.86078 -1.20846 \n";
    {
        std::ofstream outfile(filename_out);
        outfile << data;
    }

    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(filename_out, pcd, {"auto", false, false, true});
    EXPECT_FALSE(pcd.HasPointAttr("intensity"));
}

namespace {
// Test ply file containig 3DGS data
const char test_3dgs_ply_data[] = R"(ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
property float f_rest_2
property float f_rest_3
property float f_rest_4
property float f_rest_5
property float f_rest_6
property float f_rest_7
property float f_rest_8
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
0.7236  -0.52572 -0.44721  0.48902 -0.48306 -0.7263  -1.20846 0.45058 -0.98568 -0.15648  0.03506 -0.07857  0.03506 -0.06857 0.06506 -0.03857 0.14588 -0.25489  0.56895  2.56841  0.58956  1.54784 -0.34619 -0.7938 -0.48108  0.1364
0.6598  -1.42875 -2.85722 -0.69152  0.69793 -0.18625  0.24585 -0.3305 -1.58646 -0.15865 -0.05305 -0.12865 -0.08305 -0.11865 -0.09305 0.05648 -0.28579 -0.04457  0.33395  1.58847  2.55896  0.58984 0.55591  0.27858 -0.50835 -0.59577
)";
}  // namespace

// Reading ascii and check for 3DGS attributes.
TEST(TPointCloudIO, ReadPointCloudFromPLY3DGS) {
    std::string filename_out = utility::filesystem::GetTempDirectoryPath() +
                               "/test_sample_right_3dgs_format.ply";
    {
        std::ofstream outfile(filename_out);
        outfile << test_3dgs_ply_data;
    }

    t::geometry::PointCloud pcd;
    t::io::ReadPointCloud(filename_out, pcd, {"auto", false, false, true});
    EXPECT_TRUE(pcd.HasPointAttr("positions"));
    // Checks for scale, rot, f_dc and opacity and their shapes.
    EXPECT_TRUE(pcd.IsGaussianSplat());
    EXPECT_TRUE(pcd.HasPointAttr("f_rest"));
    EXPECT_EQ(pcd.GaussianSplatGetSHOrder(), 1);
}

// Read write empty point cloud.
TEST(TPointCloudIO, ReadWriteEmptyPTS) {
    t::geometry::PointCloud pcd, pcd_read;
    std::string file_name =
            utility::filesystem::GetTempDirectoryPath() + "/test_empty.pts";
    EXPECT_TRUE(pcd.IsEmpty());
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd_read.IsEmpty());
}

// Read write pts with colors and intensities.
TEST(TPointCloudIO, ReadWritePTS) {
    t::geometry::PointCloud pcd, pcd_read, pcd_i, pcd_color;
    data::PTSPointCloud pts_point_cloud;
    EXPECT_TRUE(t::io::ReadPointCloud(pts_point_cloud.GetPath(), pcd,
                                      {"auto", false, false, true}));
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointColors().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointAttr("intensities").GetLength(), 10);
    EXPECT_EQ(pcd.GetPointColors().GetDtype(), core::UInt8);
    EXPECT_TRUE(pcd.GetPointPositions()[0].AllClose(
            core::Tensor::Init<float>({4.24644, -6.42662, -50.2146})));
    EXPECT_TRUE(pcd.GetPointColors()[0].AllClose(
            core::Tensor::Init<uint8_t>({66, 50, 83})));
    EXPECT_TRUE(pcd.GetPointAttr("intensities")[0].AllClose(
            core::Tensor::Init<float>({10})));

    // Write pointcloud and match it after read.
    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();
    std::string file_name = tmp_path + "/test_read.pts";
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd.GetPointPositions().AllClose(pcd_read.GetPointPositions()));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(pcd_read.GetPointColors()));
    EXPECT_TRUE(pcd.GetPointAttr("intensities")
                        .AllClose(pcd_read.GetPointAttr("intensities")));

    // Write pointcloud with only colors and match it after read.
    pcd_read.Clear();
    pcd_color.SetPointPositions(pcd.GetPointPositions());
    pcd_color.SetPointColors(pcd.GetPointColors());
    file_name = tmp_path + "/test_color.pts";
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd_color));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(pcd_color.GetPointPositions().AllClose(
            pcd_read.GetPointPositions()));
    EXPECT_TRUE(pcd_color.GetPointColors().AllClose(pcd_read.GetPointColors()));
    EXPECT_FALSE(pcd_read.HasPointAttr("intensities"));

    // Write pointcloud with only intensities and match it after read.
    pcd_read.Clear();
    pcd_i.SetPointPositions(pcd.GetPointPositions());
    pcd_i.SetPointAttr("intensities", pcd.GetPointAttr("intensities"));
    file_name = tmp_path + "/test_intensities.pts";
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd_i));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_TRUE(
            pcd_i.GetPointPositions().AllClose(pcd_read.GetPointPositions()));
    EXPECT_TRUE(pcd_i.GetPointAttr("intensities")
                        .AllClose(pcd_read.GetPointAttr("intensities")));
    EXPECT_FALSE(pcd_read.HasPointColors());
}

// Reading pts with intensities.
TEST(TPointCloudIO, ReadPointCloudFromPTS1) {
    t::geometry::PointCloud pcd;
    data::PTSPointCloud point_cloud_sample;
    EXPECT_TRUE(t::io::ReadPointCloud(point_cloud_sample.GetPath(), pcd,
                                      {"auto", false, false, true}));
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 10);
    EXPECT_EQ(pcd.GetPointAttr("intensities").GetLength(), 10);
}

// Check PTS color float to uint8 conversion.
TEST(TPointCloudIO, WritePTSColorConversion1) {
    t::geometry::PointCloud pcd, pcd_read;
    std::string file_name = utility::filesystem::GetTempDirectoryPath() +
                            "/test_color_conversion.pts";
    pcd.SetPointPositions(core::Tensor::Init<float>({{1, 2, 3}, {4, 5, 6}}));
    pcd.SetPointColors(
            core::Tensor::Init<float>({{-1, 0.25, 0.4}, {0, 4, 0.1}}));
    EXPECT_TRUE(t::io::WritePointCloud(file_name, pcd));
    EXPECT_TRUE(t::io::ReadPointCloud(file_name, pcd_read,
                                      {"auto", false, false, true}));
    EXPECT_EQ(pcd_read.GetPointColors().ToFlatVector<uint8_t>(),
              std::vector<uint8_t>({0, 64, 102, 0, 255, 26}));
}

// Check PTS color boolean to uint8 conversion.
TEST(TPointCloudIO, WritePTSColorConversion2) {
    // Read PointCloud from PLY file.
    t::geometry::PointCloud pcd_ply;
    data::PLYPointCloud pointcloud_ply;
    t::io::ReadPointCloud(pointcloud_ply.GetPath(), pcd_ply,
                          {"auto", false, false, true});
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

    std::string filename = utility::filesystem::GetTempDirectoryPath() +
                           "/test_npz_pointcloud.npz";
    EXPECT_TRUE(t::io::WritePointCloud(filename, pcd_ply));

    // Read from the saved pointcloud.
    t::geometry::PointCloud pcd_npz;
    t::io::ReadPointCloud(filename, pcd_npz, {"auto", false, false, true});

    for (auto &kv : pcd_ply.GetPointAttr()) {
        EXPECT_TRUE(kv.second.AllClose(pcd_npz.GetPointAttr(kv.first)));
    }
}

TEST_P(PointCloudIOPermuteDevices, WriteDeviceTestPLY) {
    core::Device device = GetParam();
    std::string filename =
            utility::filesystem::GetTempDirectoryPath() + "/test_write.ply";
    core::Tensor points = core::Tensor::Ones({10, 3}, core::Float32, device);
    t::geometry::PointCloud pcd(points);
    EXPECT_TRUE(t::io::WritePointCloud(filename, pcd));
}

// Test Reading ASCII 3DGS ply, writing to binary ply and reading back.
TEST_P(PointCloudIOPermuteDevices, ReadWrite3DGSPointCloudPLY) {
    std::string filename_3dgs_ascii =
            utility::filesystem::GetTempDirectoryPath() +
            "/test_sample_3dgs_ascii.ply";
    std::string filename_3dgs_binary =
            utility::filesystem::GetTempDirectoryPath() +
            "/test_sample_3dgs_binary.ply";
    {
        std::ofstream outfile(filename_3dgs_ascii);
        outfile << test_3dgs_ply_data;
    }

    // Read PointCloud from PLY file.
    t::geometry::PointCloud pcd_ply, pcd_ply_binary;
    t::io::ReadPointCloud(filename_3dgs_ascii, pcd_ply,
                          {"auto", false, false, true});
    EXPECT_TRUE(pcd_ply.IsGaussianSplat());
    EXPECT_EQ(pcd_ply.GaussianSplatGetSHOrder(), 1);
    auto num_gaussians_base = pcd_ply.GetPointPositions().GetLength();
    EXPECT_EQ(num_gaussians_base, 2);

    EXPECT_TRUE(t::io::WritePointCloud(filename_3dgs_binary, pcd_ply,
                                       {"auto", false, false, true}));
    EXPECT_TRUE(t::io::ReadPointCloud(filename_3dgs_binary, pcd_ply_binary,
                                      {"auto", false, false, true}));

    auto num_gaussians_new = pcd_ply_binary.GetPointPositions().GetLength();
    EXPECT_EQ(num_gaussians_base, num_gaussians_new);
    AllCloseOrShow(pcd_ply.GetPointPositions(),
                   pcd_ply_binary.GetPointPositions(), 1e-5, 1e-8);
    AllCloseOrShow(pcd_ply.GetPointAttr("scale"),
                   pcd_ply_binary.GetPointAttr("scale"), 1e-5, 1e-8);
    AllCloseOrShow(pcd_ply.GetPointAttr("opacity"),
                   pcd_ply_binary.GetPointAttr("opacity"), 1e-5, 1e-8);
    AllCloseOrShow(pcd_ply.GetPointAttr("rot"),
                   pcd_ply_binary.GetPointAttr("rot"), 1e-5, 1e-8);
    AllCloseOrShow(pcd_ply.GetPointAttr("f_dc"),
                   pcd_ply_binary.GetPointAttr("f_dc"), 1e-5, 1e-8);
    AllCloseOrShow(pcd_ply.GetPointAttr("f_rest"),
                   pcd_ply_binary.GetPointAttr("f_rest"), 1e-5, 1e-8);

    auto opacity = pcd_ply.GetPointAttr("opacity");
    // Error if the shape of the attribute is not 2D with len = num_points
    pcd_ply.SetPointAttr("opacity", opacity.Reshape({num_gaussians_base}));
    EXPECT_ANY_THROW(t::io::WritePointCloud(filename_3dgs_binary, pcd_ply,
                                            {"auto", false, false, true}));
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
    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();
    std::string filename_ascii = tmp_path + "/test_pcd_pointcloud_ascii.pcd";

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_ascii, input_pcd,
            {/*ascii*/ true, /*compressed*/ false, false}));

    t::geometry::PointCloud ascii_pcd;
    t::io::ReadPointCloud(filename_ascii, ascii_pcd);

    for (auto &kv : input_pcd.GetPointAttr()) {
        EXPECT_TRUE(kv.second.AllClose(ascii_pcd.GetPointAttr(kv.first), 1e-3,
                                       1e-4));
    }

    // PCD IO for Binary format.
    std::string filename_binary = tmp_path + "/test_pcd_pointcloud_binary.pcd";

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_binary, input_pcd,
            {/*ascii*/ false, /*compressed*/ false, false}));

    t::geometry::PointCloud binary_pcd;
    t::io::ReadPointCloud(filename_binary, binary_pcd);

    for (auto &kv : input_pcd.GetPointAttr()) {
        EXPECT_TRUE(kv.second.AllClose(binary_pcd.GetPointAttr(kv.first), 1e-3,
                                       1e-4));
    }

    // PCD IO for Binary Compressed format.
    std::string filename_binary_compressed =
            tmp_path + "/test_pcd_pointcloud_binary_compressed.pcd";

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_binary_compressed, input_pcd,
            {/*ascii*/ false, /*compressed*/ true, false}));

    t::geometry::PointCloud binary_compressed_pcd;
    t::io::ReadPointCloud(filename_binary_compressed, binary_compressed_pcd);

    for (auto &kv : input_pcd.GetPointAttr()) {
        EXPECT_TRUE(kv.second.AllClose(
                binary_compressed_pcd.GetPointAttr(kv.first), 1e-3, 1e-4));
    }

    // Colors data type will be converted to UInt8 during Write / Read.
    // Only Float32, Float64, UInt8, UInt16, UInt32 colors data types are
    // supported for conversion.

    // PointCloud with Float32 type white color
    core::Tensor color_float32 = core::Tensor::Ones(
            {input_pcd.GetPointPositions().GetLength(), 3}, core::Float32);
    input_pcd.SetPointColors(color_float32);

    std::string filename_ascii_f32 =
            tmp_path + "/test_pcd_pointcloud_binary_f32_colors.pcd";

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_ascii_f32, input_pcd,
            {/*ascii*/ true, /*compressed*/ false, false}));

    t::geometry::PointCloud ascii_f32_pcd;
    t::io::ReadPointCloud(filename_ascii_f32, ascii_f32_pcd);

    core::Tensor color_uint8 =
            color_float32.To(core::Dtype::UInt8)
                    .Mul(std::numeric_limits<uint8_t>::max());

    EXPECT_TRUE(ascii_f32_pcd.GetPointColors().AllClose(color_uint8));

    // PointCloud with UInt32 type white color
    core::Tensor color_uint32 =
            color_float32.To(core::Dtype::UInt32)
                    .Mul(std::numeric_limits<uint32_t>::max());
    input_pcd.SetPointColors(color_uint32);

    std::string filename_ascii_uint32 =
            tmp_path + "/test_pcd_pointcloud_binary_uint32_colors.pcd";

    EXPECT_TRUE(t::io::WritePointCloud(
            filename_ascii_uint32, input_pcd,
            {/*ascii*/ true, /*compressed*/ false, false}));

    t::geometry::PointCloud ascii_uint32_pcd;
    t::io::ReadPointCloud(filename_ascii_uint32, ascii_uint32_pcd);

    EXPECT_TRUE(ascii_f32_pcd.GetPointColors().AllClose(color_uint8));
}

// Test 3DGS ply to splat conversion
TEST(TPointCloudIO, ReadWrite3DGSPLYToSPLAT) {
    std::string filename_ply = utility::filesystem::GetTempDirectoryPath() +
                               "/test_sample_3dgs_format.ply";
    std::string filename_splat = utility::filesystem::GetTempDirectoryPath() +
                                 "/test_sample_3dgs_format.splat";
    {
        std::ofstream outfile(filename_ply);
        outfile << test_3dgs_ply_data;
    }

    t::geometry::PointCloud pcd_ply, pcd_splat;
    t::io::ReadPointCloud(filename_ply, pcd_ply, {"auto", false, false, true});
    t::io::WritePointCloud(filename_splat, pcd_ply,
                           {"auto", false, false, true});
    t::io::ReadPointCloud(filename_splat, pcd_splat,
                          {"auto", false, false, true});

    AllCloseOrShow(pcd_ply.GetPointPositions(), pcd_splat.GetPointPositions(),
                   1e-5, 1e-8);
    AllCloseOrShow(pcd_ply.GetPointAttr("scale"),
                   pcd_splat.GetPointAttr("scale"), 1e-5, 1e-8);
    AllCloseOrShow(pcd_ply.GetPointAttr("opacity"),
                   pcd_splat.GetPointAttr("opacity"), 0,
                   0.01);  // expect quantization errors
    AllCloseOrShow(pcd_ply.GetPointAttr("rot"), pcd_splat.GetPointAttr("rot"),
                   0,
                   0.01);  // expect quantization errors
    AllCloseOrShow(pcd_ply.GetPointAttr("f_dc"), pcd_splat.GetPointAttr("f_dc"),
                   0,
                   0.01);  // expect quantization errors
    EXPECT_FALSE(pcd_splat.HasPointAttr("f_rest"));
}

// Test consistency of values after writing and reading.
TEST(TPointCloudIO, ReadWrite3DGSSPLAT) {
    t::geometry::PointCloud pcd_base;
    t::geometry::PointCloud pcd_new;

    std::string filename =
            utility::filesystem::GetTempDirectoryPath() + "/test_read.splat";
    std::string new_filename = utility::filesystem::GetTempDirectoryPath() +
                               "/new_test_read.splat";

    // Write a small splat file.  This is the same point cloud as
    // test_3dgs_ply_data (without f_rest), converted to splat using the
    // reference code from
    // github.com/antimatter15/splat/blob/367a9439609d043f1b23a9b455a77a977f2e7758/convert.py
    // (March 2025). Converted to C array with:
    // xxd -i test_3dgs_data.splat > test_3dgs_data_splat.h
    // clang-format off
    const unsigned char output_splat[64] = {
            0xd9, 0x3d, 0x39, 0x3f, 0x96, 0x95, 0x06, 0xbf, 0xb6, 0xf8, 0xe4, 0xbe, 0x96, 0xb8, 0x50, 0x41,
            0x16, 0xcf, 0xe6, 0x3f, 0x16, 0x71, 0x96, 0x40, 0x29, 0xa0, 0x39, 0xa3, 0x54, 0x1a, 0x42, 0x91,
            0xa7, 0xe8, 0x28, 0x3f, 0x48, 0xe1, 0xb6, 0xbf, 0xb1, 0xdc, 0x36, 0xc0, 0x18, 0xae, 0x9c, 0x40,
            0x08, 0xc2, 0x4e, 0x41, 0xa2, 0xdf, 0xe6, 0x3f, 0x91, 0x68, 0x0d, 0x95, 0xc7, 0xa4, 0x3f, 0x34};
    // clang-format on
    {
        std::ofstream outfile(filename,
                              std::ios::binary);  // Open in binary mode
        outfile.write(reinterpret_cast<const char *>(output_splat),
                      sizeof(output_splat));
    }

    EXPECT_TRUE(t::io::ReadPointCloudFromSPLAT(filename, pcd_base,
                                               {"splat", false, false, true}));
    EXPECT_TRUE(pcd_base.IsGaussianSplat());
    EXPECT_EQ(pcd_base.GaussianSplatGetSHOrder(), 0);
    auto num_gaussians_base = pcd_base.GetPointPositions().GetLength();
    EXPECT_EQ(num_gaussians_base, 2);

    EXPECT_TRUE(t::io::WritePointCloudToSPLAT(new_filename, pcd_base,
                                              {"splat", false, false, true}));
    EXPECT_TRUE(t::io::ReadPointCloudFromSPLAT(new_filename, pcd_new,
                                               {"splat", false, false, true}));

    auto num_gaussians_new = pcd_new.GetPointPositions().GetLength();
    EXPECT_EQ(num_gaussians_base, num_gaussians_new);
    AllCloseOrShow(pcd_base.GetPointPositions(), pcd_new.GetPointPositions(),
                   1e-5, 1e-8);
    AllCloseOrShow(pcd_base.GetPointAttr("scale"),
                   pcd_new.GetPointAttr("scale"), 1e-5, 1e-8);
    AllCloseOrShow(pcd_base.GetPointAttr("opacity"),
                   pcd_new.GetPointAttr("opacity"), 0,
                   0.01);  // expect quantization errors
    AllCloseOrShow(pcd_base.GetPointAttr("rot"), pcd_new.GetPointAttr("rot"), 0,
                   0.01);  // expect quantization errors
    AllCloseOrShow(pcd_base.GetPointAttr("f_dc"), pcd_new.GetPointAttr("f_dc"),
                   0,
                   0.01);  // expect quantization errors
}

}  // namespace tests
}  // namespace open3d
