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

#include "open3d/tgeometry/TriangleMesh.h"

#include "core/CoreTest.h"
#include "open3d/core/TensorList.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class TriangleMeshPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TriangleMesh,
                         TriangleMeshPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TriangleMeshPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        TriangleMesh,
        TriangleMeshPermuteDevicePairs,
        testing::ValuesIn(TriangleMeshPermuteDevicePairs::TestCases()));

TEST_P(TriangleMeshPermuteDevices, DefaultConstructor) {
    tgeometry::TriangleMesh mesh;

    // Inherited from Geometry3D.
    EXPECT_EQ(mesh.GetGeometryType(),
              tgeometry::Geometry::GeometryType::TriangleMesh);
    EXPECT_EQ(mesh.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(mesh.IsEmpty());
    EXPECT_FALSE(mesh.HasPoints());
    EXPECT_FALSE(mesh.HasPointColors());
    EXPECT_FALSE(mesh.HasPointNormals());
    EXPECT_FALSE(mesh.HasTriangles());
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Default dtypes.
    EXPECT_EQ(mesh.GetPoints().GetDevice(), core::Device("CPU:0"));
    EXPECT_EQ(mesh.GetPoints().GetDtype(), core::Dtype::Float32);
    EXPECT_EQ(mesh.GetTriangles().GetDevice(), core::Device("CPU:0"));
    EXPECT_EQ(mesh.GetTriangles().GetDtype(), core::Dtype::Int64);
}

TEST_P(TriangleMeshPermuteDevices, ConstructFromPoints) {
    core::Device device = GetParam();

    // Prepare data.
    core::Tensor points =
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device);
    core::Tensor single_point =
            core::Tensor::Ones({3}, core::Dtype::Float32, device);

    core::Tensor triangles =
            core::Tensor::Ones({10, 3}, core::Dtype::Int64, device);
    core::Tensor single_triangle =
            core::Tensor::Ones({3}, core::Dtype::Int64, device);

    // Copied, push back okay.
    tgeometry::TriangleMesh mesh(
            core::TensorList::FromTensor(points, /*inplace=*/false),
            core::TensorList::FromTensor(triangles, /*inplace=*/false));

    EXPECT_TRUE(mesh.HasPoints());
    EXPECT_EQ(mesh.GetPoints().GetSize(), 10);
    mesh.GetPoints().PushBack(single_point);
    EXPECT_EQ(mesh.GetPoints().GetSize(), 11);

    EXPECT_TRUE(mesh.HasTriangles());
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 10);
    mesh.GetTriangles().PushBack(single_triangle);
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 11);

    // Inplace tensorlist: cannot push_back.
    mesh = tgeometry::TriangleMesh(
            core::TensorList::FromTensor(points, /*inplace=*/true),
            core::TensorList::FromTensor(triangles, /*inplace=*/true));

    EXPECT_TRUE(mesh.HasPoints());
    EXPECT_EQ(mesh.GetPoints().GetSize(), 10);
    EXPECT_ANY_THROW(mesh.GetPoints().PushBack(single_point));

    EXPECT_TRUE(mesh.HasTriangles());
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 10);
    EXPECT_ANY_THROW(mesh.GetTriangles().PushBack(single_triangle));
}

TEST_P(TriangleMeshPermuteDevices, SynchronizedPushBack) {
    core::Device device = GetParam();

    // Create trianglemesh.
    core::TensorList points = core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device));
    core::TensorList point_colors = core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device) * 0.5);
    core::TensorList triangles = core::TensorList::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Int64, device));
    core::TensorList triangle_normals = core::TensorList::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device) * 0.5);

    tgeometry::TriangleMesh mesh(points, triangles);
    mesh.SetPointColors(point_colors);
    mesh.SetTriangleNormals(triangle_normals);

    // Exception cases are tested in TensorListMap::SynchronizedPushBack().
    std::unordered_map<std::string, core::Tensor> point_struct;
    EXPECT_EQ(mesh.GetPoints().GetSize(), 10);
    EXPECT_EQ(mesh.GetPointColors().GetSize(), 10);
    mesh.PointSynchronizedPushBack({
            {"points", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
            {"colors", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
    });
    EXPECT_EQ(mesh.GetPoints().GetSize(), 11);
    EXPECT_EQ(mesh.GetPointColors().GetSize(), 11);

    std::unordered_map<std::string, core::Tensor> triangle_struct;
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 5);
    EXPECT_EQ(mesh.GetTriangleNormals().GetSize(), 5);
    mesh.TriangleSynchronizedPushBack({
            {"triangles", core::Tensor::Ones({3}, core::Dtype::Int64, device)},
            {"normals", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
    });
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 6);
    EXPECT_EQ(mesh.GetTriangleNormals().GetSize(), 6);
}

TEST_P(TriangleMeshPermuteDevices, Getters) {
    core::Device device = GetParam();

    auto points = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device));
    auto point_colors = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2);
    auto point_labels = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3);

    auto triangles = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device));
    auto triangle_normals = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2);
    auto triangle_labels = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3);

    tgeometry::TriangleMesh mesh(points, triangles);
    mesh.SetPointColors(point_colors);
    mesh.SetPointAttr("labels", point_labels);
    mesh.SetTriangleNormals(triangle_normals);
    mesh.SetTriangleAttr("labels", triangle_labels);

    EXPECT_TRUE(mesh.GetPoints().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.GetPointColors().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetPointAttr("labels").AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3));
    EXPECT_ANY_THROW(mesh.GetPointNormals());

    EXPECT_TRUE(mesh.GetTriangles().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device)));
    EXPECT_TRUE(mesh.GetTriangleNormals().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetTriangleAttr("labels").AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3));

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetPoints(); (void)tl);
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetPointColors();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetPointAttr("labels");
                    (void)tl);
    EXPECT_ANY_THROW(const core::TensorList& tl = mesh.GetPointNormals();
                     (void)tl);

    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetTriangles(); (void)tl);
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetTriangleNormals();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetTriangleAttr("labels");
                    (void)tl);
}

TEST_P(TriangleMeshPermuteDevices, Setters) {
    core::Device device = GetParam();

    // Setters are already tested in Getters' unit tests. Here we test that
    // mismatched device should throw an exception. This test is only effective
    // is device is a CUDA device.
    tgeometry::TriangleMesh mesh(core::Dtype::Float32, core::Dtype::Float64,
                                 device);
    core::Device cpu_device = core::Device("CPU:0");
    if (cpu_device != device) {
        core::TensorList cpu_points = core::TensorList::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device));
        core::TensorList cpu_colors = core::TensorList::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device) *
                2);
        core::TensorList cpu_labels = core::TensorList::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device) *
                3);

        EXPECT_ANY_THROW(mesh.SetPoints(cpu_points));
        EXPECT_ANY_THROW(mesh.SetPointColors(cpu_colors));
        EXPECT_ANY_THROW(mesh.SetPointAttr("labels", cpu_labels));
    }
}

TEST_P(TriangleMeshPermuteDevices, Has) {
    core::Device device = GetParam();

    tgeometry::TriangleMesh mesh(core::Dtype::Float32, core::Dtype::Float64,
                                 device);
    EXPECT_FALSE(mesh.HasPoints());
    EXPECT_FALSE(mesh.HasPointColors());
    EXPECT_FALSE(mesh.HasPointNormals());
    EXPECT_FALSE(mesh.HasPointAttr("labels"));
    EXPECT_FALSE(mesh.HasTriangles());
    EXPECT_FALSE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleAttr("labels"));

    mesh.SetPoints(core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasPoints());
    mesh.SetTriangles(core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Int64, device)));
    EXPECT_TRUE(mesh.HasTriangles());

    // Different size.
    mesh.SetPointColors(core::TensorList::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device)));
    EXPECT_FALSE(mesh.HasPointColors());
    mesh.SetTriangleNormals(core::TensorList::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device)));
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Same size.
    mesh.SetPointColors(core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasPointColors());
    mesh.SetTriangleNormals(core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasTriangleNormals());
}

}  // namespace tests
}  // namespace open3d
