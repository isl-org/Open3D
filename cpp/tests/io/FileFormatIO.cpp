// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/FileFormatIO.h"

#include <fstream>

#include "open3d/io/ModelIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/rendering/Model.h"
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
    const std::string path = utility::filesystem::GetTempDirectoryPath() +
                             "/file_type_points.ply";
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

// Minimal USDA: one triangle, UsdPreviewSurface material (no textures), UV
// primvar, material binding under a scoped Materials branch and defaultPrim
// Xform.
const char kUSD[] = R"(#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def "Materials"
    {
        def Material "LitMat"
        {
            token outputs:surface.connect = </Root/Materials/LitMat/Shader.outputs:surface>
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.6, 0.9)
                token outputs:surface
            }
        }
    }
    def Mesh "Triangle"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(-0.5, 0, 0), (0.5, 0, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (0.5, 1)] (
            interpolation = "varying"
        )
        rel material:binding = </Root/Materials/LitMat>
    }
})";

TEST(FileFormatIO, ReadTriangleModelMinimalUSD) {
    const std::string path = utility::filesystem::GetTempDirectoryPath() +
                             "/minimal_triangle.usda";
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << kUSD;
    output.close();

    const auto geometry = io::ReadFileGeometryType(path);
    EXPECT_TRUE(HasGeometry(geometry, io::CONTAINS_TRIANGLES))
            << "ReadFileGeometryTypeUSD should report CONTAINS_TRIANGLES";
    EXPECT_TRUE(HasGeometry(geometry, io::CONTAINS_POINTS))
            << "ReadFileGeometryTypeUSD should report CONTAINS_POINTS";
    EXPECT_FALSE(HasGeometry(geometry, io::CONTAINS_GAUSSIAN_SPLATS))
            << "ReadFileGeometryTypeUSD should NOT report "
               "CONTAINS_GAUSSIAN_SPLATS";
    visualization::rendering::TriangleMeshModel model;
    const bool ok = io::ReadTriangleModel(path, model);
    ASSERT_TRUE(ok) << "ReadTriangleModel failed for " << path;
    ASSERT_GE(model.meshes_.size(), 1u)
            << "Assimp USD import returned no meshes (see Open3D WARNING "
               "log for ASSIMP/tinyusdz errors)";
    ASSERT_GE(model.materials_.size(), 1u);
    EXPECT_EQ(model.meshes_[0].mesh->triangles_.size(), 1u);
    EXPECT_EQ(model.meshes_[0].mesh->vertices_.size(), 3u);

    const auto& mat = model.materials_[model.meshes_[0].material_idx];
    EXPECT_EQ(mat.name, "LitMat");
    EXPECT_EQ(mat.shader, "defaultLit");
    EXPECT_FALSE(mat.has_alpha);
    EXPECT_NEAR(mat.base_color.x(), 0.2f, 1e-5f);
    EXPECT_NEAR(mat.base_color.y(), 0.6f, 1e-5f);
    EXPECT_NEAR(mat.base_color.z(), 0.9f, 1e-5f);
    EXPECT_NEAR(mat.base_color.w(), 1.0f, 1e-5f);
}

}  // namespace tests
}  // namespace open3d