// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/FileFormatIO.h"

#include <fstream>

#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {
namespace {

bool HasGeometry(io::FileGeometry geometry, io::FileGeometry flag) {
    return (int(geometry) & int(flag)) != 0;
}

const char kPointPly[] = R"(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
end_header
0 0 0
)";

const char kGaussianSplatPly[] = R"(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
0 0 0 1 1 1 1 1 0 0 0 0.1 0.2 0.3
)";

const char kMissingOpacityGaussianSplatPly[] = R"(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
0 0 0 1 1 1 1 0 0 0 0.1 0.2 0.3
)";

}  // namespace

TEST(FileFormatIO, ReadFileGeometryTypePLYPointCloud) {
    const std::string path =
            utility::filesystem::GetTempDirectoryPath() + "/file_type_points.ply";
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << kPointPly;
    output.close();

    const auto geometry = io::ReadFileGeometryType(path);
    EXPECT_TRUE(HasGeometry(geometry, io::CONTAINS_POINTS));
    EXPECT_FALSE(HasGeometry(geometry, io::CONTAINS_GAUSSIAN_SPLATS));
}

TEST(FileFormatIO, ReadFileGeometryTypePLYGaussianSplat) {
    const std::string path = utility::filesystem::GetTempDirectoryPath() +
                             "/file_type_gaussian_splat.ply";
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << kGaussianSplatPly;
    output.close();

    const auto geometry = io::ReadFileGeometryType(path);
    EXPECT_TRUE(HasGeometry(geometry, io::CONTAINS_POINTS));
    EXPECT_TRUE(HasGeometry(geometry, io::CONTAINS_GAUSSIAN_SPLATS));
}

TEST(FileFormatIO, ReadFileGeometryTypePLYRequiresFullGaussianSplatCore) {
    const std::string path = utility::filesystem::GetTempDirectoryPath() +
                             "/file_type_incomplete_gaussian_splat.ply";
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << kMissingOpacityGaussianSplatPly;
    output.close();

    const auto geometry = io::ReadFileGeometryType(path);
    EXPECT_TRUE(HasGeometry(geometry, io::CONTAINS_POINTS));
    EXPECT_FALSE(HasGeometry(geometry, io::CONTAINS_GAUSSIAN_SPLATS));
}

TEST(FileFormatIO, ReadFileGeometryTypeSPLAT) {
    const std::string path =
            utility::filesystem::GetTempDirectoryPath() + "/file_type.splat";
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write("", 0);
    output.close();

    const auto geometry = io::ReadFileGeometryType(path);
    EXPECT_TRUE(HasGeometry(geometry, io::CONTAINS_POINTS));
    EXPECT_TRUE(HasGeometry(geometry, io::CONTAINS_GAUSSIAN_SPLATS));
}

}  // namespace tests
}  // namespace open3d