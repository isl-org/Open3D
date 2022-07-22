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

#include "open3d/t/geometry/TriangleMesh.h"

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/TensorCheck.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class TriangleMeshPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TriangleMesh,
                         TriangleMeshPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TriangleMeshPermuteDevices, DefaultConstructor) {
    t::geometry::TriangleMesh mesh;

    // Inherited from Geometry3D.
    EXPECT_EQ(mesh.GetGeometryType(),
              t::geometry::Geometry::GeometryType::TriangleMesh);
    EXPECT_EQ(mesh.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(mesh.IsEmpty());
    EXPECT_FALSE(mesh.HasVertexPositions());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasTriangleIndices());
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Default device.
    EXPECT_EQ(mesh.GetDevice(), core::Device("CPU:0"));
}

TEST_P(TriangleMeshPermuteDevices, ConstructFromVertices) {
    core::Device device = GetParam();

    // Prepare data.
    core::Tensor vertices = core::Tensor::Ones({10, 3}, core::Float32, device);
    core::Tensor single_vertex = core::Tensor::Ones({3}, core::Float32, device);

    core::Tensor triangles = core::Tensor::Ones({10, 3}, core::Int64, device);
    core::Tensor single_triangle = core::Tensor::Ones({3}, core::Int64, device);

    t::geometry::TriangleMesh mesh(vertices, triangles);

    EXPECT_TRUE(mesh.HasVertexPositions());
    EXPECT_EQ(mesh.GetVertexPositions().GetLength(), 10);
    EXPECT_EQ(mesh.GetTriangleIndices().GetLength(), 10);
}

TEST_P(TriangleMeshPermuteDevices, Getters) {
    core::Device device = GetParam();

    core::Tensor vertices = core::Tensor::Ones({2, 3}, core::Float32, device);
    core::Tensor vertex_colors =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor vertex_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    core::Tensor triangles = core::Tensor::Ones({2, 3}, core::Int64, device);
    core::Tensor triangle_normals =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor triangle_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetVertexAttr("labels", vertex_labels);
    mesh.SetTriangleNormals(triangle_normals);
    mesh.SetTriangleAttr("labels", triangle_labels);

    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device)));
    EXPECT_TRUE(mesh.GetVertexColors().AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetVertexAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3));
    EXPECT_ANY_THROW(mesh.GetVertexNormals());

    EXPECT_TRUE(mesh.GetTriangleIndices().AllClose(
            core::Tensor::Ones({2, 3}, core::Int64, device)));
    EXPECT_TRUE(mesh.GetTriangleNormals().AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetTriangleAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3));

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertexPositions();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertexColors(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertexAttr("labels");
                    (void)tl);
    EXPECT_ANY_THROW(const core::Tensor& tl = mesh.GetVertexNormals();
                     (void)tl);

    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangleIndices();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangleNormals();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangleAttr("labels");
                    (void)tl);
}

TEST_P(TriangleMeshPermuteDevices, ToString) {
    core::Device device = GetParam();

    core::Tensor vertices = core::Tensor::Ones({2, 3}, core::Float32, device);
    core::Tensor vertex_colors =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor vertex_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    core::Tensor triangles = core::Tensor::Ones({2, 3}, core::Int64, device);
    core::Tensor triangle_normals =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor triangle_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetTriangleNormals(triangle_normals);

    std::string text =
            "TriangleMesh on " + device.ToString() +
            " [2 vertices (Float32) and 2 triangles (Int64)]."
            "\nVertex Attributes: colors (dtype = Float32, shape = {2, 3})."
            "\nTriangle Attributes: normals (dtype = Float32, shape = {2, "
            "3}).";

    EXPECT_STREQ(mesh.ToString().c_str(), text.c_str());

    mesh.RemoveVertexAttr("colors");
    mesh.RemoveTriangleAttr("normals");

    // Mesh with only primary attributes.
    std::string text_2 = "TriangleMesh on " + device.ToString() +
                         " [2 vertices (Float32) and 2 triangles (Int64)]."
                         "\nVertex Attributes: None."
                         "\nTriangle Attributes: None.";

    EXPECT_STREQ(mesh.ToString().c_str(), text_2.c_str());

    // Empty mesh.
    mesh.Clear();
    std::string text_3 = "TriangleMesh on " + device.ToString() +
                         " [0 vertices and 0 triangles].";
    EXPECT_STREQ(mesh.ToString().c_str(), text_3.c_str());
}

TEST_P(TriangleMeshPermuteDevices, Setters) {
    core::Device device = GetParam();

    // Setters are already tested in Getters' unit tests. Here we test that
    // mismatched device should throw an exception. This test is only effective
    // if device is a CUDA device.
    t::geometry::TriangleMesh mesh(device);
    core::Device cpu_device = core::Device("CPU:0");
    if (cpu_device != device) {
        core::Tensor cpu_vertices =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device);
        core::Tensor cpu_colors =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device) * 2;
        core::Tensor cpu_labels =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device) * 3;

        EXPECT_ANY_THROW(mesh.SetVertexPositions(cpu_vertices));
        EXPECT_ANY_THROW(mesh.SetVertexColors(cpu_colors));
        EXPECT_ANY_THROW(mesh.SetVertexAttr("labels", cpu_labels));
    }
}

TEST_P(TriangleMeshPermuteDevices, RemoveAttr) {
    core::Device device = GetParam();

    core::Tensor vertices = core::Tensor::Ones({2, 3}, core::Float32, device);
    core::Tensor vertex_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    core::Tensor triangles = core::Tensor::Ones({2, 3}, core::Int64, device);
    core::Tensor triangle_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    t::geometry::TriangleMesh mesh(vertices, triangles);

    mesh.SetVertexAttr("labels", vertex_labels);

    EXPECT_NO_THROW(mesh.GetVertexAttr("labels"));
    mesh.RemoveVertexAttr("labels");
    EXPECT_ANY_THROW(mesh.GetVertexAttr("labels"));

    mesh.SetTriangleAttr("labels", triangle_labels);

    EXPECT_NO_THROW(mesh.GetTriangleAttr("labels"));
    mesh.RemoveTriangleAttr("labels");
    EXPECT_ANY_THROW(mesh.GetTriangleAttr("labels"));

    // Not allowed to delete primary key attribute.
    EXPECT_ANY_THROW(mesh.RemoveVertexAttr("positions"));
    EXPECT_ANY_THROW(mesh.RemoveTriangleAttr("indices"));
}

TEST_P(TriangleMeshPermuteDevices, Has) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    EXPECT_FALSE(mesh.HasVertexPositions());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasVertexAttr("labels"));
    EXPECT_FALSE(mesh.HasTriangleIndices());
    EXPECT_FALSE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleAttr("labels"));

    mesh.SetVertexPositions(core::Tensor::Ones({10, 3}, core::Float32, device));
    EXPECT_TRUE(mesh.HasVertexPositions());
    mesh.SetTriangleIndices(core::Tensor::Ones({10, 3}, core::Int64, device));
    EXPECT_TRUE(mesh.HasTriangleIndices());

    // Different size.
    mesh.SetVertexColors(core::Tensor::Ones({5, 3}, core::Float32, device));
    EXPECT_FALSE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(core::Tensor::Ones({5, 3}, core::Float32, device));
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Same size.
    mesh.SetVertexColors(core::Tensor::Ones({10, 3}, core::Float32, device));
    EXPECT_TRUE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(core::Tensor::Ones({10, 3}, core::Float32, device));
    EXPECT_TRUE(mesh.HasTriangleNormals());
}

TEST_P(TriangleMeshPermuteDevices, Transform) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    core::Tensor transformation = core::Tensor::Init<float>(
            {{1, 1, 0, 1}, {0, 1, 1, 1}, {0, 1, 0, 1}, {0, 0, 0, 1}}, device);

    mesh.SetVertexPositions(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));
    mesh.SetVertexNormals(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));

    mesh.Transform(transformation);
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Init<float>({{3, 3, 2}, {3, 3, 2}}, device)));
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Init<float>({{2, 2, 1}, {2, 2, 1}}, device)));
}

TEST_P(TriangleMeshPermuteDevices, Translate) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    core::Tensor translation = core::Tensor::Init<float>({10, 20, 30}, device);

    // Relative.
    mesh.SetVertexPositions(
            core::Tensor::Init<float>({{0, 1, 2}, {6, 7, 8}}, device));
    mesh.SetVertexNormals(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));

    mesh.Translate(translation, /*relative=*/true);

    // Normals do not translate.
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device)));

    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Init<float>({{10, 21, 32}, {16, 27, 38}}, device)));

    // Non-relative.
    mesh.SetVertexPositions(
            core::Tensor::Init<float>({{0, 1, 2}, {6, 7, 8}}, device));
    mesh.Translate(translation, /*relative=*/false);

    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Init<float>({{7, 17, 27}, {13, 23, 33}}, device)));
}

TEST_P(TriangleMeshPermuteDevices, Scale) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    core::Tensor rotation = core::Tensor::Init<float>(
            {{1, 1, 0}, {0, 1, 1}, {0, 1, 0}}, device);
    core::Tensor center = core::Tensor::Ones({3}, core::Dtype::Float32, device);
    double scale = 4;

    mesh.SetVertexPositions(core::Tensor::Init<float>(
            {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}, device));

    mesh.Scale(scale, center);
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(core::Tensor::Init<float>(
            {{-3, -3, -3}, {1, 1, 1}, {5, 5, 5}}, device)));
}

TEST_P(TriangleMeshPermuteDevices, Rotate) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    core::Tensor rotation = core::Tensor::Init<float>(
            {{1, 1, 0}, {0, 1, 1}, {0, 1, 0}}, device);
    core::Tensor center = core::Tensor::Ones({3}, core::Dtype::Float32, device);

    mesh.SetVertexPositions(
            core::Tensor::Init<float>({{2, 2, 2}, {2, 2, 2}}, device));
    mesh.SetVertexNormals(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));

    mesh.Rotate(rotation, center);
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Init<float>({{3, 3, 2}, {3, 3, 2}}, device)));
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Init<float>({{2, 2, 1}, {2, 2, 1}}, device)));
}

TEST_P(TriangleMeshPermuteDevices, NormalizeNormals) {
    core::Device device = GetParam();

    std::vector<Eigen::Vector3d> ref_vertex_normals = {
            {0.692861, 0.323767, 0.644296}, {0.650010, 0.742869, 0.160101},
            {0.379563, 0.870761, 0.312581}, {0.575046, 0.493479, 0.652534},
            {0.320665, 0.448241, 0.834418}, {0.691127, 0.480526, 0.539850},
            {0.227557, 0.973437, 0.025284}, {0.281666, 0.156994, 0.946582},
            {0.341869, 0.894118, 0.289273}, {0.103335, 0.972118, 0.210498},
            {0.441745, 0.723783, 0.530094}, {0.336903, 0.727710, 0.597441},
            {0.434917, 0.862876, 0.257471}, {0.636619, 0.435239, 0.636619},
            {0.393717, 0.876213, 0.277918}, {0.275051, 0.633543, 0.723167},
            {0.061340, 0.873191, 0.483503}, {0.118504, 0.276510, 0.953677},
            {0.930383, 0.360677, 0.065578}, {0.042660, 0.989719, 0.136513},
            {0.175031, 0.720545, 0.670953}, {0.816905, 0.253392, 0.518130},
            {0.377967, 0.767871, 0.517219}, {0.782281, 0.621223, 0.046017},
            {0.314385, 0.671253, 0.671253}};

    std::vector<Eigen::Vector3d> ref_triangle_normals = {
            {0.331843, 0.660368, 0.673642},  {0.920309, 0.198342, 0.337182},
            {0.778098, 0.279317, 0.562624},  {0.547237, 0.723619, 0.420604},
            {0.360898, 0.671826, 0.646841},  {0.657733, 0.738934, 0.146163},
            {0.92945, 0.0241415, 0.368159},  {0.160811, 0.969595, 0.18446},
            {0.922633, 0.298499, 0.244226},  {0.874092, 0.189272, 0.44737},
            {0.776061, 0.568382, 0.273261},  {0.663812, 0.544981, 0.5122},
            {0.763905, 0.22794, 0.603732},   {0.518555, 0.758483, 0.394721},
            {0.892885, 0.283206, 0.350074},  {0.657978, 0.751058, 0.054564},
            {0.872328, 0.483025, 0.0756979}, {0.170605, 0.588415, 0.790356},
            {0.982336, 0.178607, 0.0558146}, {0.881626, 0.121604, 0.456013},
            {0.616413, 0.573987, 0.539049},  {0.372896, 0.762489, 0.528733},
            {0.669715, 0.451103, 0.589905},  {0.771164, 0.0571233, 0.634068},
            {0.620625, 0.620625, 0.479217}};

    const core::Tensor ref_vertex_normals_tensor =
            core::eigen_converter::EigenVector3dVectorToTensor(
                    ref_vertex_normals, core::Dtype::Float64, device);
    const core::Tensor ref_triangle_normals_tensor =
            core::eigen_converter::EigenVector3dVectorToTensor(
                    ref_triangle_normals, core::Dtype::Float64, device);

    size_t size = 25;
    std::vector<Eigen::Vector3d> vertex_normals, triangle_normals, vertices;
    std::vector<Eigen::Vector3i> triangles;
    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);
    vertex_normals.resize(size);
    triangle_normals.resize(size);
    vertices.resize(size);
    Rand(vertex_normals, dmin, dmax, 0);
    Rand(triangle_normals, dmin, dmax, 1);
    Rand(vertices, dmin, dmax, 2);
    for (size_t i = 0; i < size; i++)
        triangles.push_back(Eigen::Vector3i(i, (i + 1) % size, (i + 2) % size));

    t::geometry::TriangleMesh t_mesh(
            core::eigen_converter::EigenVector3dVectorToTensor(
                    vertices, core::Dtype::Float64, device),
            core::eigen_converter::EigenVector3iVectorToTensor(
                    triangles, core::Dtype::Int64, device));
    t_mesh.SetVertexNormals(core::eigen_converter::EigenVector3dVectorToTensor(
            vertex_normals, core::Dtype::Float64, device));
    t_mesh.SetTriangleNormals(
            core::eigen_converter::EigenVector3dVectorToTensor(
                    triangle_normals, core::Dtype::Float64, device));

    t_mesh.NormalizeNormals();

    EXPECT_TRUE(t_mesh.GetVertexNormals().AllClose(ref_vertex_normals_tensor));
    EXPECT_TRUE(
            t_mesh.GetTriangleNormals().AllClose(ref_triangle_normals_tensor));
}

TEST_P(TriangleMeshPermuteDevices, ComputeTriangleNormals) {
    core::Device device = GetParam();

    std::vector<Eigen::Vector3d> ref = {{-0.119231, 0.738792, 0.663303},
                                        {-0.115181, 0.730934, 0.672658},
                                        {-0.589738, -0.764139, -0.261344},
                                        {-0.330250, 0.897644, -0.291839},
                                        {-0.164192, 0.976753, 0.137819},
                                        {-0.475702, 0.727947, 0.493762},
                                        {0.990884, -0.017339, -0.133596},
                                        {0.991673, 0.091418, -0.090700},
                                        {0.722410, 0.154580, -0.673965},
                                        {0.598552, -0.312929, -0.737435},
                                        {0.712875, -0.628251, -0.311624},
                                        {0.233815, -0.638800, -0.732984},
                                        {0.494773, -0.472428, -0.729391},
                                        {0.583861, 0.796905, 0.155075},
                                        {-0.277650, -0.948722, -0.151119},
                                        {-0.791337, 0.093176, 0.604238},
                                        {0.569287, 0.158108, 0.806793},
                                        {0.115315, 0.914284, 0.388314},
                                        {0.105421, 0.835841, -0.538754},
                                        {0.473326, 0.691900, -0.545195},
                                        {0.719515, 0.684421, -0.117757},
                                        {-0.713642, -0.691534, -0.111785},
                                        {-0.085377, -0.916925, 0.389820},
                                        {0.787892, 0.611808, -0.070127},
                                        {0.788022, 0.488544, 0.374628}};

    size_t size = 25;

    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;
    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    vertices.resize(size);
    Rand(vertices, dmin, dmax, 0);

    for (size_t i = 0; i < size; i++)
        triangles.push_back(Eigen::Vector3i(i, (i + 1) % size, (i + 2) % size));

    t::geometry::TriangleMesh t_mesh(
            core::eigen_converter::EigenVector3dVectorToTensor(
                    vertices, core::Dtype::Float64, device),
            core::eigen_converter::EigenVector3iVectorToTensor(
                    triangles, core::Dtype::Int64, device));

    const core::Tensor ref_tensor =
            core::eigen_converter::EigenVector3dVectorToTensor(
                    ref, core::Dtype::Float64, device);
    t_mesh.ComputeTriangleNormals();

    EXPECT_TRUE(t_mesh.GetTriangleNormals().AllClose(ref_tensor));
}

TEST_P(TriangleMeshPermuteDevices, ComputeVertexNormals) {
    core::Device device = GetParam();

    std::vector<Eigen::Vector3d> ref = {{0.635868, 0.698804, 0.327636},
                                        {0.327685, 0.717012, 0.615237},
                                        {-0.346072, 0.615418, 0.708163},
                                        {-0.690485, 0.663544, 0.287993},
                                        {-0.406664, 0.913428, -0.0165494},
                                        {-0.356568, 0.888296, 0.289466},
                                        {-0.276491, 0.894931, 0.350216},
                                        {0.262855, 0.848183, 0.459883},
                                        {0.933461, 0.108347, -0.341923},
                                        {0.891804, 0.0506666, -0.449577},
                                        {0.735392, -0.110383, -0.668592},
                                        {0.469090, -0.564602, -0.679102},
                                        {0.418223, -0.628547, -0.655757},
                                        {0.819226, 0.168537, -0.548145},
                                        {0.963613, 0.103044, -0.246642},
                                        {-0.506244, 0.320837, 0.800488},
                                        {0.122226, -0.0580307, 0.990804},
                                        {0.175502, 0.533543, 0.827364},
                                        {0.384132, 0.892338, 0.237015},
                                        {0.273664, 0.896739, -0.347804},
                                        {0.36153, 0.784805, -0.503366},
                                        {0.4297, 0.646636, -0.630253},
                                        {-0.264834, -0.96397, -0.0250043},
                                        {0.940214, -0.336158, -0.0547315},
                                        {0.86265, 0.449603, 0.231714}};

    size_t size = 25;

    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;
    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    vertices.resize(size);
    Rand(vertices, dmin, dmax, 0);

    for (size_t i = 0; i < size; i++)
        triangles.push_back(Eigen::Vector3i(i, (i + 1) % size, (i + 2) % size));

    t::geometry::TriangleMesh t_mesh(
            core::eigen_converter::EigenVector3dVectorToTensor(
                    vertices, core::Dtype::Float64, device),
            core::eigen_converter::EigenVector3iVectorToTensor(
                    triangles, core::Dtype::Int64, device));

    const core::Tensor ref_tensor =
            core::eigen_converter::EigenVector3dVectorToTensor(
                    ref, core::Dtype::Float64, device);
    t_mesh.ComputeVertexNormals();
    EXPECT_TRUE(t_mesh.GetVertexNormals().AllClose(ref_tensor));
}

TEST_P(TriangleMeshPermuteDevices, FromLegacy) {
    core::Device device = GetParam();
    geometry::TriangleMesh legacy_mesh;
    legacy_mesh.vertices_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)};
    legacy_mesh.vertex_colors_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, 1)};
    legacy_mesh.vertex_normals_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(2, 2, 2), Eigen::Vector3d(2, 2, 2)};
    legacy_mesh.triangles_ = std::vector<Eigen::Vector3i>{
            Eigen::Vector3i(3, 3, 3), Eigen::Vector3i(3, 3, 3)};
    legacy_mesh.triangle_normals_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(4, 4, 4), Eigen::Vector3d(4, 4, 4)};
    legacy_mesh.triangle_uvs_ = std::vector<Eigen::Vector2d>{
            Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(0.2, 0.3),
            Eigen::Vector2d(0.4, 0.5), Eigen::Vector2d(0.6, 0.7),
            Eigen::Vector2d(0.8, 0.9), Eigen::Vector2d(1.0, 1.1)};

    core::Dtype float_dtype = core::Float32;
    core::Dtype int_dtype = core::Int64;
    t::geometry::TriangleMesh mesh = t::geometry::TriangleMesh::FromLegacy(
            legacy_mesh, float_dtype, int_dtype, device);

    EXPECT_TRUE(mesh.HasVertexPositions());
    EXPECT_TRUE(mesh.HasVertexColors());
    EXPECT_TRUE(mesh.HasVertexNormals());
    EXPECT_TRUE(mesh.HasTriangleIndices());
    EXPECT_TRUE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleColors());
    EXPECT_TRUE(mesh.HasTriangleAttr("texture_uvs"));
    EXPECT_FALSE(mesh.HasVertexAttr("texture_uvs"));

    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetVertexPositions(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetVertexPositions(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetVertexColors(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetVertexNormals(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetTriangleIndices(), int_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetTriangleNormals(), float_dtype));
    EXPECT_NO_THROW(core::AssertTensorDtype(mesh.GetTriangleAttr("texture_uvs"),
                                            float_dtype));

    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 0));
    EXPECT_TRUE(mesh.GetVertexColors().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 1));
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 2));
    EXPECT_TRUE(mesh.GetTriangleIndices().AllClose(
            core::Tensor::Ones({2, 3}, int_dtype, device) * 3));
    EXPECT_TRUE(mesh.GetTriangleNormals().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 4));
    EXPECT_TRUE(mesh.GetTriangleAttr("texture_uvs")
                        .AllClose(core::Tensor::Arange(0., 1.1, 0.1,
                                                       float_dtype, device)
                                          .Reshape({-1, 3, 2})));
}

TEST_P(TriangleMeshPermuteDevices, ToLegacy) {
    using ::testing::ElementsAreArray;
    using ::testing::FloatEq;
    using ::testing::Pointwise;
    core::Device device = GetParam();

    core::Dtype float_dtype = core::Float32;
    core::Dtype int_dtype = core::Int64;

    t::geometry::TriangleMesh mesh(device);
    mesh.SetVertexPositions(core::Tensor::Ones({2, 3}, float_dtype, device) *
                            0);
    mesh.SetVertexColors(core::Tensor::Ones({2, 3}, float_dtype, device) * 1);
    mesh.SetVertexNormals(core::Tensor::Ones({2, 3}, float_dtype, device) * 2);
    mesh.SetTriangleIndices(core::Tensor::Ones({2, 3}, int_dtype, device) * 3);
    mesh.SetTriangleNormals(core::Tensor::Ones({2, 3}, float_dtype, device) *
                            4);
    mesh.SetTriangleAttr("texture_uvs",
                         core::Tensor::Arange(0., 1.1, 0.1, float_dtype, device)
                                 .Reshape({-1, 3, 2}));

    geometry::TriangleMesh legacy_mesh = mesh.ToLegacy();
    EXPECT_EQ(legacy_mesh.vertices_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)}));
    EXPECT_EQ(legacy_mesh.vertex_colors_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, 1)}));
    EXPECT_EQ(legacy_mesh.vertex_normals_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(2, 2, 2), Eigen::Vector3d(2, 2, 2)}));
    EXPECT_EQ(legacy_mesh.triangles_,
              std::vector<Eigen::Vector3i>(
                      {Eigen::Vector3i(3, 3, 3), Eigen::Vector3i(3, 3, 3)}));
    EXPECT_EQ(legacy_mesh.triangle_normals_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(4, 4, 4), Eigen::Vector3d(4, 4, 4)}));
    EXPECT_THAT(legacy_mesh.triangle_uvs_,
                ElementsAreArray({Pointwise(FloatEq(), {0.0, 0.1}),
                                  Pointwise(FloatEq(), {0.2, 0.3}),
                                  Pointwise(FloatEq(), {0.4, 0.5}),
                                  Pointwise(FloatEq(), {0.6, 0.7}),
                                  Pointwise(FloatEq(), {0.8, 0.9}),
                                  Pointwise(FloatEq(), {1.0, 1.1})}));
}

TEST_P(TriangleMeshPermuteDevices, CreateBox) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with default parameters.
    t::geometry::TriangleMesh box_default =
            t::geometry::TriangleMesh::CreateBox();

    core::Tensor vertex_positions_default =
            core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                       {1.0, 0.0, 0.0},
                                       {0.0, 0.0, 1.0},
                                       {1.0, 0.0, 1.0},
                                       {0.0, 1.0, 0.0},
                                       {1.0, 1.0, 0.0},
                                       {0.0, 1.0, 1.0},
                                       {1.0, 1.0, 1.0}});

    core::Tensor triangle_indices_default =
            core::Tensor::Init<int64_t>({{4, 7, 5},
                                         {4, 6, 7},
                                         {0, 2, 4},
                                         {2, 6, 4},
                                         {0, 1, 2},
                                         {1, 3, 2},
                                         {1, 5, 7},
                                         {1, 7, 3},
                                         {2, 3, 7},
                                         {2, 7, 6},
                                         {0, 4, 1},
                                         {1, 4, 5}});

    EXPECT_TRUE(box_default.GetVertexPositions().AllClose(
            vertex_positions_default));
    EXPECT_TRUE(box_default.GetTriangleIndices().AllClose(
            triangle_indices_default));

    // Test with custom parameters.
    t::geometry::TriangleMesh box_custom = t::geometry::TriangleMesh::CreateBox(
            2, 3, 4, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                        {2.0, 0.0, 0.0},
                                        {0.0, 0.0, 4.0},
                                        {2.0, 0.0, 4.0},
                                        {0.0, 3.0, 0.0},
                                        {2.0, 3.0, 0.0},
                                        {0.0, 3.0, 4.0},
                                        {2.0, 3.0, 4.0}},
                                       device);

    core::Tensor triangle_indices_custom =
            core::Tensor::Init<int32_t>({{4, 7, 5},
                                         {4, 6, 7},
                                         {0, 2, 4},
                                         {2, 6, 4},
                                         {0, 1, 2},
                                         {1, 3, 2},
                                         {1, 5, 7},
                                         {1, 7, 3},
                                         {2, 3, 7},
                                         {2, 7, 6},
                                         {0, 4, 1},
                                         {1, 4, 5}},
                                        device);

    EXPECT_TRUE(
            box_custom.GetVertexPositions().AllClose(vertex_positions_custom));
    EXPECT_TRUE(
            box_custom.GetTriangleIndices().AllClose(triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateSphere) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh sphere_custom =
            t::geometry::TriangleMesh::CreateSphere(1, 3, float_dtype_custom,
                                                    int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 1.0},
                                        {0.0, 0.0, -1.0},
                                        {0.866025, 0, 0.5},
                                        {0.433013, 0.75, 0.5},
                                        {-0.433013, 0.75, 0.5},
                                        {-0.866025, 0.0, 0.5},
                                        {-0.433013, -0.75, 0.5},
                                        {0.433013, -0.75, 0.5},
                                        {0.866025, 0.0, -0.5},
                                        {0.433013, 0.75, -0.5},
                                        {-0.433013, 0.75, -0.5},
                                        {-0.866025, 0.0, -0.5},
                                        {-0.433013, -0.75, -0.5},
                                        {0.433013, -0.75, -0.5}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 2, 3},   {1, 9, 8},   {0, 3, 4},   {1, 10, 9}, {0, 4, 5},
             {1, 11, 10}, {0, 5, 6},   {1, 12, 11}, {0, 6, 7},  {1, 13, 12},
             {0, 7, 2},   {1, 8, 13},  {8, 3, 2},   {8, 9, 3},  {9, 4, 3},
             {9, 10, 4},  {10, 5, 4},  {10, 11, 5}, {11, 6, 5}, {11, 12, 6},
             {12, 7, 6},  {12, 13, 7}, {13, 2, 7},  {13, 8, 2}},
            device);

    EXPECT_TRUE(sphere_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(sphere_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateTetrahedron) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh tetrahedron_custom =
            t::geometry::TriangleMesh::CreateTetrahedron(
                    2, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{1.88562, 0.0, -0.666667},
                                        {-0.942809, 1.63299, -0.666667},
                                        {-0.942809, -1.63299, -0.666667},
                                        {0.0, 0.0, 2}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 2, 1}, {0, 3, 2}, {0, 1, 3}, {1, 2, 3}}, device);
    EXPECT_TRUE(tetrahedron_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(tetrahedron_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateOctahedron) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh octahedron_custom =
            t::geometry::TriangleMesh::CreateOctahedron(
                    2, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{2.0, 0.0, 0.0},
                                        {0.0, 2.0, 0.0},
                                        {0.0, 0.0, 2.0},
                                        {-2.0, 0.0, 0.0},
                                        {0.0, -2.0, 0.0},
                                        {0.0, 0.0, -2.0}},
                                       device);

    core::Tensor triangle_indices_custom =
            core::Tensor::Init<int32_t>({{0, 1, 2},
                                         {1, 3, 2},
                                         {3, 4, 2},
                                         {4, 0, 2},
                                         {0, 5, 1},
                                         {1, 5, 3},
                                         {3, 5, 4},
                                         {4, 5, 0}},
                                        device);
    EXPECT_TRUE(octahedron_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(octahedron_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateIcosahedron) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh icosahedron_custom =
            t::geometry::TriangleMesh::CreateIcosahedron(
                    2, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{-2.0, 0.0, 3.23607},
                                        {2.0, 0.0, 3.23607},
                                        {2.0, 0.0, -3.23607},
                                        {-2.0, 0.0, -3.23607},
                                        {0.0, -3.23607, 2.0},
                                        {0.0, 3.23607, 2.0},
                                        {0.0, 3.23607, -2.0},
                                        {0.0, -3.23607, -2.0},
                                        {-3.23607, -2.0, 0.0},
                                        {3.23607, -2.0, 0.0},
                                        {3.23607, 2.0, 0.0},
                                        {-3.23607, 2.0, 0.0}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 4, 1},  {0, 1, 5},  {1, 4, 9},  {1, 9, 10}, {1, 10, 5},
             {0, 8, 4},  {0, 11, 8}, {0, 5, 11}, {5, 6, 11}, {5, 10, 6},
             {4, 8, 7},  {4, 7, 9},  {3, 6, 2},  {3, 2, 7},  {2, 6, 10},
             {2, 10, 9}, {2, 9, 7},  {3, 11, 6}, {3, 8, 11}, {3, 7, 8}},
            device);
    EXPECT_TRUE(icosahedron_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(icosahedron_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateCylinder) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh cylinder_custom =
            t::geometry::TriangleMesh::CreateCylinder(
                    1, 2, 3, 3, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 1.0},
                                        {0.0, 0.0, -1.0},
                                        {1.0, 0.0, 1.0},
                                        {-0.5, 0.866025, 1.0},
                                        {-0.5, -0.866025, 1.0},
                                        {1.0, 0.0, 0.333333},
                                        {-0.5, 0.866025, 0.333333},
                                        {-0.5, -0.866025, 0.333333},
                                        {1.0, 0.0, -0.333333},
                                        {-0.5, 0.866025, -0.333333},
                                        {-0.5, -0.866025, -0.333333},
                                        {1.0, 0.0, -1.0},
                                        {-0.5, 0.866025, -1.0},
                                        {-0.5, -0.866025, -1.0}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 2, 3},   {1, 12, 11},  {0, 3, 4},   {1, 13, 12}, {0, 4, 2},
             {1, 11, 13}, {5, 3, 2},    {5, 6, 3},   {6, 4, 3},   {6, 7, 4},
             {7, 2, 4},   {7, 5, 2},    {8, 6, 5},   {8, 9, 6},   {9, 7, 6},
             {9, 10, 7},  {10, 5, 7},   {10, 8, 5},  {11, 9, 8},  {11, 12, 9},
             {12, 10, 9}, {12, 13, 10}, {13, 8, 10}, {13, 11, 8}},
            device);
    EXPECT_TRUE(cylinder_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(cylinder_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateCone) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh cone_custom =
            t::geometry::TriangleMesh::CreateCone(
                    2, 4, 3, 2, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                        {0.0, 0.0, 4.0},
                                        {2.0, 0.0, 0.0},
                                        {-1.0, 1.73205, 0.0},
                                        {-1.0, -1.73205, 0.0},
                                        {1.0, 0.0, 2.0},
                                        {-0.5, 0.866025, 2},
                                        {-0.5, -0.866025, 2}},
                                       device);

    core::Tensor triangle_indices_custom =
            core::Tensor::Init<int32_t>({{0, 3, 2},
                                         {1, 5, 6},
                                         {0, 4, 3},
                                         {1, 6, 7},
                                         {0, 2, 4},
                                         {1, 7, 5},
                                         {6, 2, 3},
                                         {6, 5, 2},
                                         {7, 3, 4},
                                         {7, 6, 3},
                                         {5, 4, 2},
                                         {5, 7, 4}},
                                        device);
    EXPECT_TRUE(
            cone_custom.GetVertexPositions().AllClose(vertex_positions_custom));
    EXPECT_TRUE(
            cone_custom.GetTriangleIndices().AllClose(triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateTorus) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh torus_custom =
            t::geometry::TriangleMesh::CreateTorus(
                    2, 1, 6, 3, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{3.0, 0.0, 0.0},
                                        {1.5, 0.0, 0.866025},
                                        {1.5, 0.0, -0.866025},
                                        {1.5, 2.59808, 0.0},
                                        {0.75, 1.29904, 0.866025},
                                        {0.75, 1.29904, -0.866025},
                                        {-1.5, 2.59808, 0},
                                        {-0.75, 1.29904, 0.866025},
                                        {-0.75, 1.29904, -0.866025},
                                        {-3.0, 0.0, 0.0},
                                        {-1.5, 0.0, 0.866025},
                                        {-1.5, 0.0, -0.866025},
                                        {-1.5, -2.59808, 0.0},
                                        {-0.75, -1.29904, 0.866025},
                                        {-0.75, -1.29904, -0.866025},
                                        {1.5, -2.59808, 0.0},
                                        {0.75, -1.29904, 0.866025},
                                        {0.75, -1.29904, -0.866025}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{3, 4, 0},    {0, 4, 1},    {4, 5, 1},    {1, 5, 2},
             {5, 3, 2},    {2, 3, 0},    {6, 7, 3},    {3, 7, 4},
             {7, 8, 4},    {4, 8, 5},    {8, 6, 5},    {5, 6, 3},
             {9, 10, 6},   {6, 10, 7},   {10, 11, 7},  {7, 11, 8},
             {11, 9, 8},   {8, 9, 6},    {12, 13, 9},  {9, 13, 10},
             {13, 14, 10}, {10, 14, 11}, {14, 12, 11}, {11, 12, 9},
             {15, 16, 12}, {12, 16, 13}, {16, 17, 13}, {13, 17, 14},
             {17, 15, 14}, {14, 15, 12}, {0, 1, 15},   {15, 1, 16},
             {1, 2, 16},   {16, 2, 17},  {2, 0, 17},   {17, 0, 15}},
            device);
    EXPECT_TRUE(torus_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(torus_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateArrow) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh arrow_custom =
            t::geometry::TriangleMesh::CreateArrow(1, 2, 4, 2, 4, 1, 1,
                                                   float_dtype_custom,
                                                   int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 4.0},
                                        {0.0, 0.0, 0.0},
                                        {1.0, 0.0, 4.0},
                                        {0.0, 1.0, 4.0},
                                        {-1.0, 0.0, 4.0},
                                        {0.0, -1.0, 4.0},
                                        {1.0, 0.0, 0.0},
                                        {0.0, 1.0, 0.0},
                                        {-1.0, 0.0, 0.0},
                                        {0.0, -1.0, 0.0},
                                        {0.0, 0.0, 4.0},
                                        {0.0, 0.0, 6.0},
                                        {2.0, 0.0, 4.0},
                                        {0.0, 2.0, 4.0},
                                        {-2.0, 0.0, 4.0},
                                        {0.0, -2.0, 4.0}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 2, 3},    {1, 7, 6},    {0, 3, 4},    {1, 8, 7},
             {0, 4, 5},    {1, 9, 8},    {0, 5, 2},    {1, 6, 9},
             {6, 3, 2},    {6, 7, 3},    {7, 4, 3},    {7, 8, 4},
             {8, 5, 4},    {8, 9, 5},    {9, 2, 5},    {9, 6, 2},
             {10, 13, 12}, {11, 12, 13}, {10, 14, 13}, {11, 13, 14},
             {10, 15, 14}, {11, 14, 15}, {10, 12, 15}, {11, 15, 12}},
            device);
    EXPECT_TRUE(arrow_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(arrow_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateMobius) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh mobius_custom =
            t::geometry::TriangleMesh::CreateMobius(10, 2, 1, 1, 1, 1, 1,
                                                    float_dtype_custom,
                                                    int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.5, 0.0, 0.0},
                                        {1.5, 0.0, 0.0},
                                        {0.424307, 0.308277, -0.154508},
                                        {1.19373, 0.867294, 0.154508},
                                        {0.184017, 0.566346, -0.293893},
                                        {0.434017, 1.33577, 0.293893},
                                        {-0.218199, 0.671548, -0.404508},
                                        {-0.399835, 1.23057, 0.404508},
                                        {-0.684017, 0.496967, -0.475528},
                                        {-0.934017, 0.678603, 0.475528},
                                        {-1.0, 0.0, -0.5},
                                        {-1.0, 0.0, 0.5},
                                        {-0.934017, -0.678603, -0.475528},
                                        {-0.684017, -0.496967, 0.475528},
                                        {-0.399835, -1.23057, -0.404508},
                                        {-0.218199, -0.671548, 0.404508},
                                        {0.434017, -1.33577, -0.293893},
                                        {0.184017, -0.566346, 0.293893},
                                        {1.19373, -0.867294, -0.154508},
                                        {0.424307, -0.308277, 0.154508}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 3, 1},    {0, 2, 3},    {3, 2, 4},    {3, 4, 5},
             {4, 7, 5},    {4, 6, 7},    {7, 6, 8},    {7, 8, 9},
             {8, 11, 9},   {8, 10, 11},  {11, 10, 12}, {11, 12, 13},
             {12, 15, 13}, {12, 14, 15}, {15, 14, 16}, {15, 16, 17},
             {16, 19, 17}, {16, 18, 19}, {18, 19, 1},  {1, 19, 0}},
            device);
    EXPECT_TRUE(mobius_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(mobius_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

}  // namespace tests
}  // namespace open3d
