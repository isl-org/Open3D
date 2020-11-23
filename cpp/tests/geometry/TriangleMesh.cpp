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

#include "open3d/geometry/TriangleMesh.h"

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/PointCloud.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

void ExpectMeshEQ(const open3d::geometry::TriangleMesh& mesh0,
                  const open3d::geometry::TriangleMesh& mesh1,
                  double threshold = 1e-6) {
    ExpectEQ(mesh0.vertices_, mesh1.vertices_, threshold);
    ExpectEQ(mesh0.vertex_normals_, mesh1.vertex_normals_, threshold);
    ExpectEQ(mesh0.vertex_colors_, mesh1.vertex_colors_, threshold);
    ExpectEQ(mesh0.triangles_, mesh1.triangles_);
    ExpectEQ(mesh0.triangle_normals_, mesh1.triangle_normals_, threshold);
}

TEST(TriangleMesh, Constructor) {
    geometry::TriangleMesh tm;

    // inherited from Geometry2D
    EXPECT_EQ(geometry::Geometry::GeometryType::TriangleMesh,
              tm.GetGeometryType());
    EXPECT_EQ(3, tm.Dimension());

    // public member variables
    EXPECT_EQ(0u, tm.vertices_.size());
    EXPECT_EQ(0u, tm.vertex_normals_.size());
    EXPECT_EQ(0u, tm.vertex_colors_.size());
    EXPECT_EQ(0u, tm.triangles_.size());
    EXPECT_EQ(0u, tm.triangle_normals_.size());

    // public members
    EXPECT_TRUE(tm.IsEmpty());

    ExpectEQ(Zero3d, tm.GetMinBound());
    ExpectEQ(Zero3d, tm.GetMaxBound());

    EXPECT_FALSE(tm.HasVertices());
    EXPECT_FALSE(tm.HasVertexNormals());
    EXPECT_FALSE(tm.HasVertexColors());
    EXPECT_FALSE(tm.HasTriangles());
    EXPECT_FALSE(tm.HasTriangleNormals());
}

TEST(TriangleMesh, DISABLED_MemberData) { NotImplemented(); }

TEST(TriangleMesh, Clear) {
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.vertex_colors_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    Rand(tm.vertices_, dmin, dmax, 0);
    Rand(tm.vertex_normals_, dmin, dmax, 0);
    Rand(tm.vertex_colors_, dmin, dmax, 0);
    Rand(tm.triangles_, imin, imax, 0);
    Rand(tm.triangle_normals_, dmin, dmax, 0);

    EXPECT_FALSE(tm.IsEmpty());

    ExpectEQ(Eigen::Vector3d(19.607843, 0.0, 0.0), tm.GetMinBound());
    ExpectEQ(Eigen::Vector3d(996.078431, 996.078431, 996.078431),
             tm.GetMaxBound());

    EXPECT_TRUE(tm.HasVertices());
    EXPECT_TRUE(tm.HasVertexNormals());
    EXPECT_TRUE(tm.HasVertexColors());
    EXPECT_TRUE(tm.HasTriangles());
    EXPECT_TRUE(tm.HasTriangleNormals());

    tm.Clear();

    // public members
    EXPECT_TRUE(tm.IsEmpty());

    ExpectEQ(Zero3d, tm.GetMinBound());
    ExpectEQ(Zero3d, tm.GetMaxBound());

    EXPECT_FALSE(tm.HasVertices());
    EXPECT_FALSE(tm.HasVertexNormals());
    EXPECT_FALSE(tm.HasVertexColors());
    EXPECT_FALSE(tm.HasTriangles());
    EXPECT_FALSE(tm.HasTriangleNormals());
}

TEST(TriangleMesh, IsEmpty) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_TRUE(tm.IsEmpty());

    tm.vertices_.resize(size);

    EXPECT_FALSE(tm.IsEmpty());
}

TEST(TriangleMesh, GetMinBound) {
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    ExpectEQ(Eigen::Vector3d(19.607843, 0.0, 0.0), tm.GetMinBound());
}

TEST(TriangleMesh, GetMaxBound) {
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    ExpectEQ(Eigen::Vector3d(996.078431, 996.078431, 996.078431),
             tm.GetMaxBound());
}

TEST(TriangleMesh, Transform) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {1.411252, 4.274168, 3.130918}, {1.231757, 4.154505, 3.183678},
            {1.403168, 4.268779, 2.121679}, {1.456767, 4.304511, 2.640845},
            {1.620902, 4.413935, 1.851255}, {1.374684, 4.249790, 3.062485},
            {1.328160, 4.218773, 1.795728}, {1.713446, 4.475631, 1.860145},
            {1.409239, 4.272826, 2.011462}, {1.480169, 4.320113, 1.177780}};

    std::vector<Eigen::Vector3d> ref_vertex_normals = {
            {396.470588, 1201.176471, 880.352941},
            {320.392157, 1081.176471, 829.019608},
            {268.627451, 817.647059, 406.666667},
            {338.431373, 1000.392157, 614.117647},
            {423.137255, 1152.549020, 483.607843},
            {432.549020, 1337.647059, 964.392157},
            {139.607843, 443.921569, 189.176471},
            {291.764706, 762.352941, 317.058824},
            {134.117647, 407.058824, 191.882353},
            {274.509804, 801.568627, 218.627451}};

    std::vector<Eigen::Vector3d> ref_triangle_normals = {
            {396.470588, 1201.176471, 880.352941},
            {320.392157, 1081.176471, 829.019608},
            {268.627451, 817.647059, 406.666667},
            {338.431373, 1000.392157, 614.117647},
            {423.137255, 1152.549020, 483.607843},
            {432.549020, 1337.647059, 964.392157},
            {139.607843, 443.921569, 189.176471},
            {291.764706, 762.352941, 317.058824},
            {134.117647, 407.058824, 191.882353},
            {274.509804, 801.568627, 218.627451}};

    int size = 10;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.triangle_normals_.resize(size);

    Rand(tm.vertices_, dmin, dmax, 0);
    Rand(tm.vertex_normals_, dmin, dmax, 0);
    Rand(tm.triangle_normals_, dmin, dmax, 0);

    Eigen::Matrix4d transformation;
    transformation << 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16;

    tm.Transform(transformation);

    ExpectEQ(ref_vertices, tm.vertices_);
    ExpectEQ(ref_vertex_normals, tm.vertex_normals_);
    ExpectEQ(ref_triangle_normals, tm.triangle_normals_);
}

TEST(TriangleMesh, OperatorAppend) {
    size_t size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm0;
    geometry::TriangleMesh tm1;

    tm0.vertices_.resize(size);
    tm0.vertex_normals_.resize(size);
    tm0.vertex_colors_.resize(size);
    tm0.triangles_.resize(size);
    tm0.triangle_normals_.resize(size);

    tm1.vertices_.resize(size);
    tm1.vertex_normals_.resize(size);
    tm1.vertex_colors_.resize(size);
    tm1.triangles_.resize(size);
    tm1.triangle_normals_.resize(size);

    Rand(tm0.vertices_, dmin, dmax, 0);
    Rand(tm0.vertex_normals_, dmin, dmax, 0);
    Rand(tm0.vertex_colors_, dmin, dmax, 0);
    Rand(tm0.triangles_, imin, imax, 0);
    Rand(tm0.triangle_normals_, dmin, dmax, 0);

    Rand(tm1.vertices_, dmin, dmax, 0);
    Rand(tm1.vertex_normals_, dmin, dmax, 0);
    Rand(tm1.vertex_colors_, dmin, dmax, 1);
    Rand(tm1.triangles_, imin, imax, 0);
    Rand(tm1.triangle_normals_, dmin, dmax, 0);

    geometry::TriangleMesh tm(tm0);
    tm += tm1;

    EXPECT_EQ(2 * size, tm.vertices_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.vertices_[i], tm.vertices_[i + 0]);
        ExpectEQ(tm1.vertices_[i], tm.vertices_[i + size]);
    }

    EXPECT_EQ(2 * size, tm.vertex_normals_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.vertex_normals_[i], tm.vertex_normals_[i + 0]);
        ExpectEQ(tm1.vertex_normals_[i], tm.vertex_normals_[i + size]);
    }

    EXPECT_EQ(2 * size, tm.vertex_colors_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.vertex_colors_[i], tm.vertex_colors_[i + 0]);
        ExpectEQ(tm1.vertex_colors_[i], tm.vertex_colors_[i + size]);
    }

    // NOTE: why is this offset required only for triangles?
    EXPECT_EQ(2 * size, tm.triangles_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.triangles_[i], tm.triangles_[i + 0]);
        ExpectEQ(Eigen::Vector3i(tm1.triangles_[i](0, 0) + size,
                                 tm1.triangles_[i](1, 0) + size,
                                 tm1.triangles_[i](2, 0) + size),
                 tm.triangles_[i + size]);
    }

    EXPECT_EQ(2 * size, tm.triangle_normals_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.triangle_normals_[i], tm.triangle_normals_[i + 0]);
        ExpectEQ(tm1.triangle_normals_[i], tm.triangle_normals_[i + size]);
    }
}

TEST(TriangleMesh, OperatorADD) {
    size_t size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm0;
    geometry::TriangleMesh tm1;

    tm0.vertices_.resize(size);
    tm0.vertex_normals_.resize(size);
    tm0.vertex_colors_.resize(size);
    tm0.triangles_.resize(size);
    tm0.triangle_normals_.resize(size);

    tm1.vertices_.resize(size);
    tm1.vertex_normals_.resize(size);
    tm1.vertex_colors_.resize(size);
    tm1.triangles_.resize(size);
    tm1.triangle_normals_.resize(size);

    Rand(tm0.vertices_, dmin, dmax, 0);
    Rand(tm0.vertex_normals_, dmin, dmax, 0);
    Rand(tm0.vertex_colors_, dmin, dmax, 0);
    Rand(tm0.triangles_, imin, imax, 0);
    Rand(tm0.triangle_normals_, dmin, dmax, 0);

    Rand(tm1.vertices_, dmin, dmax, 0);
    Rand(tm1.vertex_normals_, dmin, dmax, 0);
    Rand(tm1.vertex_colors_, dmin, dmax, 1);
    Rand(tm1.triangles_, imin, imax, 0);
    Rand(tm1.triangle_normals_, dmin, dmax, 0);

    geometry::TriangleMesh tm = tm0 + tm1;

    EXPECT_EQ(2 * size, tm.vertices_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.vertices_[i], tm.vertices_[i + 0]);
        ExpectEQ(tm1.vertices_[i], tm.vertices_[i + size]);
    }

    EXPECT_EQ(2 * size, tm.vertex_normals_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.vertex_normals_[i], tm.vertex_normals_[i + 0]);
        ExpectEQ(tm1.vertex_normals_[i], tm.vertex_normals_[i + size]);
    }

    EXPECT_EQ(2 * size, tm.vertex_colors_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.vertex_colors_[i], tm.vertex_colors_[i + 0]);
        ExpectEQ(tm1.vertex_colors_[i], tm.vertex_colors_[i + size]);
    }

    // NOTE: why is this offset required only for triangles?
    EXPECT_EQ(2 * size, tm.triangles_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.triangles_[i], tm.triangles_[i + 0]);
        ExpectEQ(Eigen::Vector3i(tm1.triangles_[i](0, 0) + size,
                                 tm1.triangles_[i](1, 0) + size,
                                 tm1.triangles_[i](2, 0) + size),
                 tm.triangles_[i + size]);
    }

    EXPECT_EQ(2 * size, tm.triangle_normals_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(tm0.triangle_normals_[i], tm.triangle_normals_[i + 0]);
        ExpectEQ(tm1.triangle_normals_[i], tm.triangle_normals_[i + size]);
    }
}

TEST(TriangleMesh, ComputeTriangleNormals) {
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

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    for (size_t i = 0; i < size; i++)
        tm.triangles_.push_back(
                Eigen::Vector3i(i, (i + 1) % size, (i + 2) % size));

    tm.ComputeTriangleNormals();

    ExpectEQ(ref, tm.triangle_normals_);
}

TEST(TriangleMesh, ComputeVertexNormals) {
    std::vector<Eigen::Vector3d> ref = {
            {0.635868, 0.698804, 0.327636},    {0.327685, 0.717012, 0.615237},
            {-0.346072, 0.615418, 0.708163},   {-0.690485, 0.663544, 0.287992},
            {-0.406664, 0.913428, -0.016549},  {-0.356568, 0.888296, 0.289466},
            {-0.276491, 0.894931, 0.350216},   {0.262855, 0.848183, 0.459883},
            {0.933461, 0.108347, -0.341923},   {0.891804, 0.050667, -0.449577},
            {0.735392, -0.110383, -0.668592},  {0.469090, -0.564602, -0.679102},
            {0.418223, -0.628548, -0.655758},  {0.819226, 0.168537, -0.548145},
            {0.963613, 0.103044, -0.246642},   {-0.506244, 0.320837, 0.800488},
            {0.122226, -0.058031, 0.990804},   {0.175502, 0.533543, 0.827364},
            {0.384132, 0.892338, 0.237015},    {0.273664, 0.896739, -0.347804},
            {0.361530, 0.784805, -0.503366},   {0.429700, 0.646636, -0.630253},
            {-0.264834, -0.963970, -0.025005}, {0.940214, -0.336158, -0.054732},
            {0.862650, 0.449603, 0.231714}};

    size_t size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    for (size_t i = 0; i < size; i++)
        tm.triangles_.push_back(
                Eigen::Vector3i(i, (i + 1) % size, (i + 2) % size));

    tm.ComputeVertexNormals();

    ExpectEQ(ref, tm.vertex_normals_);
}

TEST(TriangleMesh, ComputeAdjacencyList) {
    // 4-sided pyramid with A as top vertex, bottom has two triangles
    Eigen::Vector3d A(0, 0, 1);    // 0
    Eigen::Vector3d B(1, 1, 0);    // 1
    Eigen::Vector3d C(-1, 1, 0);   // 2
    Eigen::Vector3d D(-1, -1, 0);  // 3
    Eigen::Vector3d E(1, -1, 0);   // 4
    std::vector<Eigen::Vector3d> vertices{A, B, C, D, E};

    geometry::TriangleMesh tm;
    tm.vertices_.insert(tm.vertices_.end(), std::begin(vertices),
                        std::end(vertices));
    tm.triangles_ = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(0, 2, 3),
                     Eigen::Vector3i(0, 3, 4), Eigen::Vector3i(0, 4, 1),
                     Eigen::Vector3i(1, 2, 4), Eigen::Vector3i(2, 3, 4)};
    EXPECT_FALSE(tm.HasAdjacencyList());
    tm.ComputeAdjacencyList();
    EXPECT_TRUE(tm.HasAdjacencyList());

    // A
    EXPECT_TRUE(tm.adjacency_list_[0] == std::unordered_set<int>({1, 2, 3, 4}));
    // B
    EXPECT_TRUE(tm.adjacency_list_[1] == std::unordered_set<int>({0, 2, 4}));
    // C
    EXPECT_TRUE(tm.adjacency_list_[2] == std::unordered_set<int>({0, 1, 3, 4}));
    // D
    EXPECT_TRUE(tm.adjacency_list_[3] == std::unordered_set<int>({0, 2, 4}));
    // E
    EXPECT_TRUE(tm.adjacency_list_[4] == std::unordered_set<int>({0, 1, 2, 3}));
}

TEST(TriangleMesh, Purge) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {839.215686, 392.156863, 780.392157},
            {796.078431, 909.803922, 196.078431},
            {333.333333, 764.705882, 274.509804},
            {552.941176, 474.509804, 627.450980},
            {364.705882, 509.803922, 949.019608},
            {913.725490, 635.294118, 713.725490},
            {141.176471, 603.921569, 15.686275},
            {239.215686, 133.333333, 803.921569},
            {152.941176, 400.000000, 129.411765},
            {105.882353, 996.078431, 215.686275},
            {509.803922, 835.294118, 611.764706},
            {294.117647, 635.294118, 521.568627},
            {490.196078, 972.549020, 290.196078},
            {768.627451, 525.490196, 768.627451},
            {400.000000, 890.196078, 282.352941},
            {349.019608, 803.921569, 917.647059},
            {66.666667, 949.019608, 525.490196},
            {82.352941, 192.156863, 662.745098},
            {890.196078, 345.098039, 62.745098},
            {19.607843, 454.901961, 62.745098},
            {235.294118, 968.627451, 901.960784},
            {847.058824, 262.745098, 537.254902},
            {372.549020, 756.862745, 509.803922},
            {666.666667, 529.411765, 39.215686}};

    std::vector<Eigen::Vector3d> ref_vertex_normals = {
            {839.215686, 392.156863, 780.392157},
            {796.078431, 909.803922, 196.078431},
            {333.333333, 764.705882, 274.509804},
            {552.941176, 474.509804, 627.450980},
            {364.705882, 509.803922, 949.019608},
            {913.725490, 635.294118, 713.725490},
            {141.176471, 603.921569, 15.686275},
            {239.215686, 133.333333, 803.921569},
            {152.941176, 400.000000, 129.411765},
            {105.882353, 996.078431, 215.686275},
            {509.803922, 835.294118, 611.764706},
            {294.117647, 635.294118, 521.568627},
            {490.196078, 972.549020, 290.196078},
            {768.627451, 525.490196, 768.627451},
            {400.000000, 890.196078, 282.352941},
            {349.019608, 803.921569, 917.647059},
            {66.666667, 949.019608, 525.490196},
            {82.352941, 192.156863, 662.745098},
            {890.196078, 345.098039, 62.745098},
            {19.607843, 454.901961, 62.745098},
            {235.294118, 968.627451, 901.960784},
            {847.058824, 262.745098, 537.254902},
            {372.549020, 756.862745, 509.803922},
            {666.666667, 529.411765, 39.215686}};

    std::vector<Eigen::Vector3d> ref_vertex_colors = {
            {839.215686, 392.156863, 780.392157},
            {796.078431, 909.803922, 196.078431},
            {333.333333, 764.705882, 274.509804},
            {552.941176, 474.509804, 627.450980},
            {364.705882, 509.803922, 949.019608},
            {913.725490, 635.294118, 713.725490},
            {141.176471, 603.921569, 15.686275},
            {239.215686, 133.333333, 803.921569},
            {152.941176, 400.000000, 129.411765},
            {105.882353, 996.078431, 215.686275},
            {509.803922, 835.294118, 611.764706},
            {294.117647, 635.294118, 521.568627},
            {490.196078, 972.549020, 290.196078},
            {768.627451, 525.490196, 768.627451},
            {400.000000, 890.196078, 282.352941},
            {349.019608, 803.921569, 917.647059},
            {66.666667, 949.019608, 525.490196},
            {82.352941, 192.156863, 662.745098},
            {890.196078, 345.098039, 62.745098},
            {19.607843, 454.901961, 62.745098},
            {235.294118, 968.627451, 901.960784},
            {847.058824, 262.745098, 537.254902},
            {372.549020, 756.862745, 509.803922},
            {666.666667, 529.411765, 39.215686}};

    std::vector<Eigen::Vector3i> ref_triangles = {
            {20, 9, 18},  {19, 21, 4}, {8, 18, 6}, {13, 11, 15}, {8, 12, 22},
            {21, 15, 17}, {3, 14, 0},  {5, 3, 19}, {2, 23, 5},   {12, 20, 14},
            {7, 15, 12},  {11, 23, 6}, {9, 21, 6}, {8, 19, 22},  {1, 22, 12},
            {1, 4, 15},   {21, 8, 1},  {0, 10, 1}, {5, 23, 21},  {20, 6, 12},
            {8, 18, 12},  {16, 12, 0}};

    std::vector<Eigen::Vector3d> ref_triangle_normals = {
            {839.215686, 392.156863, 780.392157},
            {796.078431, 909.803922, 196.078431},
            {333.333333, 764.705882, 274.509804},
            {552.941176, 474.509804, 627.450980},
            {364.705882, 509.803922, 949.019608},
            {913.725490, 635.294118, 713.725490},
            {141.176471, 603.921569, 15.686275},
            {239.215686, 133.333333, 803.921569},
            {105.882353, 996.078431, 215.686275},
            {509.803922, 835.294118, 611.764706},
            {294.117647, 635.294118, 521.568627},
            {490.196078, 972.549020, 290.196078},
            {400.000000, 890.196078, 282.352941},
            {349.019608, 803.921569, 917.647059},
            {66.666667, 949.019608, 525.490196},
            {82.352941, 192.156863, 662.745098},
            {890.196078, 345.098039, 62.745098},
            {19.607843, 454.901961, 62.745098},
            {235.294118, 968.627451, 901.960784},
            {847.058824, 262.745098, 537.254902},
            {372.549020, 756.862745, 509.803922},
            {666.666667, 529.411765, 39.215686}};

    int size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm0;
    geometry::TriangleMesh tm1;

    tm0.vertices_.resize(size);
    tm0.vertex_normals_.resize(size);
    tm0.vertex_colors_.resize(size);
    tm0.triangles_.resize(size);
    tm0.triangle_normals_.resize(size);

    tm1.vertices_.resize(size);
    tm1.vertex_normals_.resize(size);
    tm1.vertex_colors_.resize(size);
    tm1.triangles_.resize(size);
    tm1.triangle_normals_.resize(size);

    Rand(tm0.vertices_, dmin, dmax, 0);
    Rand(tm0.vertex_normals_, dmin, dmax, 0);
    Rand(tm0.vertex_colors_, dmin, dmax, 0);
    Rand(tm0.triangles_, imin, imax, 0);
    Rand(tm0.triangle_normals_, dmin, dmax, 0);

    Rand(tm1.vertices_, dmin, dmax, 0);
    Rand(tm1.vertex_normals_, dmin, dmax, 0);
    Rand(tm1.vertex_colors_, dmin, dmax, 1);
    Rand(tm1.triangles_, imin, imax, 0);
    Rand(tm1.triangle_normals_, dmin, dmax, 0);

    geometry::TriangleMesh tm = tm0 + tm1;

    tm.RemoveDuplicatedVertices();
    tm.RemoveDuplicatedTriangles();
    tm.RemoveUnreferencedVertices();
    tm.RemoveDegenerateTriangles();

    ExpectEQ(ref_vertices, tm.vertices_);
    ExpectEQ(ref_vertex_normals, tm.vertex_normals_);
    ExpectEQ(ref_vertex_colors, tm.vertex_colors_);
    ExpectEQ(ref_triangles, tm.triangles_);
    ExpectEQ(ref_triangle_normals, tm.triangle_normals_);
}

TEST(TriangleMesh, MergeCloseVertices) {
    geometry::TriangleMesh mesh;
    mesh.vertices_ = {{0.000000, 0.000000, 0.000000},
                      {0.000000, 0.200000, 0.000000},
                      {1.000000, 0.200000, 0.000000},
                      {1.000000, 0.000000, 0.000000}};
    mesh.vertex_normals_ = {{0.000000, 0.000000, 1.000000},
                            {0.000000, 0.000000, 1.000000},
                            {0.000000, 0.000000, 1.000000},
                            {0.000000, 0.000000, 1.000000}};
    mesh.triangles_ = {{0, 2, 1}, {2, 0, 3}};
    mesh.triangle_normals_ = {{0.000000, 0.000000, 1.000000},
                              {0.000000, 0.000000, 1.000000}};

    geometry::TriangleMesh ref;
    ref.vertices_ = {{0.000000, 0.100000, 0.000000},
                     {1.000000, 0.100000, 0.000000}};
    ref.vertex_normals_ = {{0.000000, 0.000000, 1.000000},
                           {0.000000, 0.000000, 1.000000}};
    ref.triangles_ = {{0, 1, 0}, {1, 0, 1}};
    ref.triangle_normals_ = {{0.000000, 0.000000, 0.000000},
                             {0.000000, 0.000000, -0.000000}};

    mesh.MergeCloseVertices(1);
    ExpectMeshEQ(mesh, ref);

    mesh.vertices_ = {{0.000000, 0.000000, 0.000000},
                      {0.000000, 0.200000, 0.000000},
                      {1.000000, 0.200000, 0.000000},
                      {1.000000, 0.000000, 0.000000}};
    mesh.vertex_normals_ = {{0.000000, 0.000000, 1.000000},
                            {0.000000, 0.000000, 1.000000},
                            {0.000000, 0.000000, 1.000000},
                            {0.000000, 0.000000, 1.000000}};
    mesh.triangles_ = {{0, 2, 1}, {2, 0, 3}};
    mesh.triangle_normals_ = {{0.000000, 0.000000, 1.000000},
                              {0.000000, 0.000000, 1.000000}};
    ref.vertices_ = {{0.000000, 0.000000, 0.000000},
                     {0.000000, 0.200000, 0.000000},
                     {1.000000, 0.200000, 0.000000},
                     {1.000000, 0.000000, 0.000000}};
    ref.vertex_normals_ = {{0.000000, 0.000000, 1.000000},
                           {0.000000, 0.000000, 1.000000},
                           {0.000000, 0.000000, 1.000000},
                           {0.000000, 0.000000, 1.000000}};
    ref.triangles_ = {{0, 2, 1}, {2, 0, 3}};
    ref.triangle_normals_ = {{0.000000, 0.000000, 1.000000},
                             {0.000000, 0.000000, 1.000000}};

    mesh.MergeCloseVertices(0.1);
    ExpectMeshEQ(mesh, ref);
}

TEST(TriangleMesh, SamplePointsUniformly) {
    auto mesh_empty = geometry::TriangleMesh();
    EXPECT_THROW(mesh_empty.SamplePointsUniformly(100), std::runtime_error);

    std::vector<Eigen::Vector3d> vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    std::vector<Eigen::Vector3i> triangles = {{0, 1, 2}};

    auto mesh_simple = geometry::TriangleMesh();
    mesh_simple.vertices_ = vertices;
    mesh_simple.triangles_ = triangles;

    size_t n_points = 100;
    auto pcd_simple = mesh_simple.SamplePointsUniformly(n_points);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == 0);
    EXPECT_TRUE(pcd_simple->normals_.size() == 0);

    std::vector<Eigen::Vector3d> colors = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
    std::vector<Eigen::Vector3d> normals = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
    mesh_simple.vertex_colors_ = colors;
    mesh_simple.vertex_normals_ = normals;
    pcd_simple = mesh_simple.SamplePointsUniformly(n_points);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(pcd_simple->colors_[pidx], Eigen::Vector3d(1, 0, 0));
        ExpectEQ(pcd_simple->normals_[pidx], Eigen::Vector3d(0, 1, 0));
    }

    // use triangle normal instead of the vertex normals
    EXPECT_FALSE(mesh_simple.HasTriangleNormals());
    pcd_simple = mesh_simple.SamplePointsUniformly(n_points, true);
    // the mesh now has triangle normals as a side effect.
    EXPECT_TRUE(mesh_simple.HasTriangleNormals());
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(pcd_simple->colors_[pidx], Eigen::Vector3d(1, 0, 0));
        ExpectEQ(pcd_simple->normals_[pidx], Eigen::Vector3d(0, 0, 1));
    }

    // use triangle normal, this time the mesh has no vertex normals
    mesh_simple.vertex_normals_.clear();
    pcd_simple = mesh_simple.SamplePointsUniformly(n_points, true);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(pcd_simple->colors_[pidx], Eigen::Vector3d(1, 0, 0));
        ExpectEQ(pcd_simple->normals_[pidx], Eigen::Vector3d(0, 0, 1));
    }
}

TEST(TriangleMesh, FilterSharpen) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = mesh->FilterSharpen(1, 1);
    std::vector<Eigen::Vector3d> ref1 = {
            {0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {-4, 0, 0}, {0, -4, 0}};
    ExpectEQ(mesh->vertices_, ref1);

    mesh = mesh->FilterSharpen(9, 0.1);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {42.417997, 0, 0},
                                         {0, 42.417997, 0},
                                         {-42.417997, 0, 0},
                                         {0, -42.417997, 0}};
    ExpectEQ(mesh->vertices_, ref2);
}

TEST(TriangleMesh, FilterSmoothSimple) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = mesh->FilterSmoothSimple(1);
    std::vector<Eigen::Vector3d> ref1 = {{0, 0, 0},
                                         {0.25, 0, 0},
                                         {0, 0.25, 0},
                                         {-0.25, 0, 0},
                                         {0, -0.25, 0}};
    ExpectEQ(mesh->vertices_, ref1, 1e-4);

    mesh = mesh->FilterSmoothSimple(3);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.003906, 0, 0},
                                         {0, 0.003906, 0},
                                         {-0.003906, 0, 0},
                                         {0, -0.003906, 0}};
    ExpectEQ(mesh->vertices_, ref2, 1e-4);
}

TEST(TriangleMesh, FilterSmoothLaplacian) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = mesh->FilterSmoothLaplacian(1, 0.5);
    std::vector<Eigen::Vector3d> ref1 = {
            {0, 0, 0}, {0.5, 0, 0}, {0, 0.5, 0}, {-0.5, 0, 0}, {0, -0.5, 0}};
    ExpectEQ(mesh->vertices_, ref1, 1e-3);

    mesh = mesh->FilterSmoothLaplacian(10, 0.5);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.000488, 0, 0},
                                         {0, 0.000488, 0},
                                         {-0.000488, 0, 0},
                                         {0, -0.000488, 0}};
    ExpectEQ(mesh->vertices_, ref2, 1e-3);
}

TEST(TriangleMesh, FilterSmoothTaubin) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = mesh->FilterSmoothTaubin(1, 0.5, -0.53);
    std::vector<Eigen::Vector3d> ref1 = {{0, 0, 0},
                                         {0.765, 0, 0},
                                         {0, 0.765, 0},
                                         {-0.765, 0, 0},
                                         {0, -0.765, 0}};
    ExpectEQ(mesh->vertices_, ref1, 1e-4);

    mesh = mesh->FilterSmoothTaubin(10, 0.5, -0.53);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.052514, 0, 0},
                                         {0, 0.052514, 0},
                                         {-0.052514, 0, 0},
                                         {0, -0.052514, 0}};
    ExpectEQ(mesh->vertices_, ref2, 1e-4);
}

TEST(TriangleMesh, HasVertices) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertices());

    tm.vertices_.resize(size);

    EXPECT_TRUE(tm.HasVertices());
}

TEST(TriangleMesh, HasTriangles) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangles());

    tm.vertices_.resize(size);
    tm.triangles_.resize(size);

    EXPECT_TRUE(tm.HasTriangles());
}

TEST(TriangleMesh, HasVertexNormals) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexNormals());

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);

    EXPECT_TRUE(tm.HasVertexNormals());
}

TEST(TriangleMesh, HasVertexColors) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexColors());

    tm.vertices_.resize(size);
    tm.vertex_colors_.resize(size);

    EXPECT_TRUE(tm.HasVertexColors());
}

TEST(TriangleMesh, HasTriangleNormals) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangleNormals());

    tm.vertices_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    EXPECT_TRUE(tm.HasTriangleNormals());
}

TEST(TriangleMesh, NormalizeNormals) {
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
            {0.331843, 0.660368, 0.673642}, {0.920309, 0.198342, 0.337182},
            {0.778098, 0.279317, 0.562624}, {0.547237, 0.723619, 0.420604},
            {0.360898, 0.671826, 0.646841}, {0.657733, 0.738934, 0.146163},
            {0.929450, 0.024142, 0.368159}, {0.160811, 0.969595, 0.184460},
            {0.922633, 0.298499, 0.244226}, {0.874092, 0.189272, 0.447370},
            {0.776061, 0.568382, 0.273261}, {0.663812, 0.544981, 0.512200},
            {0.763905, 0.227940, 0.603732}, {0.518555, 0.758483, 0.394721},
            {0.892885, 0.283206, 0.350074}, {0.657978, 0.751058, 0.054564},
            {0.872328, 0.483025, 0.075698}, {0.170605, 0.588415, 0.790356},
            {0.982336, 0.178607, 0.055815}, {0.881626, 0.121604, 0.456013},
            {0.616413, 0.573987, 0.539049}, {0.372896, 0.762489, 0.528733},
            {0.669715, 0.451103, 0.589905}, {0.771164, 0.057123, 0.634068},
            {0.620625, 0.620625, 0.479217}};

    int size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    geometry::TriangleMesh tm;

    tm.vertex_normals_.resize(size);
    tm.triangle_normals_.resize(size);
    Rand(tm.vertex_normals_, dmin, dmax, 0);
    Rand(tm.triangle_normals_, dmin, dmax, 1);

    tm.NormalizeNormals();

    ExpectEQ(ref_vertex_normals, tm.vertex_normals_);
    ExpectEQ(ref_triangle_normals, tm.triangle_normals_);
}

TEST(TriangleMesh, PaintUniformColor) {
    int size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_colors_.resize(size);

    Eigen::Vector3d color(233. / 255., 171. / 255., 53.0 / 255.);
    tm.PaintUniformColor(color);

    for (size_t i = 0; i < tm.vertex_colors_.size(); i++)
        ExpectEQ(color, tm.vertex_colors_[i]);
}

TEST(TriangleMesh, EulerPoincareCharacteristic) {
    EXPECT_TRUE(geometry::TriangleMesh::CreateBox()
                        ->EulerPoincareCharacteristic() == 2);
    EXPECT_TRUE(geometry::TriangleMesh::CreateSphere()
                        ->EulerPoincareCharacteristic() == 2);
    EXPECT_TRUE(geometry::TriangleMesh::CreateCylinder()
                        ->EulerPoincareCharacteristic() == 2);
    EXPECT_TRUE(geometry::TriangleMesh::CreateCone()
                        ->EulerPoincareCharacteristic() == 2);
    EXPECT_TRUE(geometry::TriangleMesh::CreateTorus()
                        ->EulerPoincareCharacteristic() == 0);

    geometry::TriangleMesh mesh0;
    mesh0.vertices_ = {{0, 0, 0},  {1, 0, 0},  {1, 1, 0}, {1, 1, 1},
                       {-1, 0, 0}, {-1, 1, 0}, {-1, 1, 1}};
    mesh0.triangles_ = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
                        {0, 4, 5}, {0, 4, 6}, {0, 5, 6}, {4, 5, 6}};
    EXPECT_TRUE(mesh0.EulerPoincareCharacteristic() == 3);

    geometry::TriangleMesh mesh1;
    mesh1.vertices_ = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 0, 2}, {1, 0.5, 1}};
    mesh1.triangles_ = {{0, 1, 2}, {1, 2, 3}, {1, 2, 4}};
    EXPECT_TRUE(mesh1.EulerPoincareCharacteristic() == 1);
}

TEST(TriangleMesh, IsEdgeManifold) {
    EXPECT_TRUE(geometry::TriangleMesh::CreateBox()->IsEdgeManifold(true));
    EXPECT_TRUE(geometry::TriangleMesh::CreateSphere()->IsEdgeManifold(true));
    EXPECT_TRUE(geometry::TriangleMesh::CreateCylinder()->IsEdgeManifold(true));
    EXPECT_TRUE(geometry::TriangleMesh::CreateCone()->IsEdgeManifold(true));
    EXPECT_TRUE(geometry::TriangleMesh::CreateTorus()->IsEdgeManifold(true));

    EXPECT_TRUE(geometry::TriangleMesh::CreateBox()->IsEdgeManifold(false));
    EXPECT_TRUE(geometry::TriangleMesh::CreateSphere()->IsEdgeManifold(false));
    EXPECT_TRUE(
            geometry::TriangleMesh::CreateCylinder()->IsEdgeManifold(false));
    EXPECT_TRUE(geometry::TriangleMesh::CreateCone()->IsEdgeManifold(false));
    EXPECT_TRUE(geometry::TriangleMesh::CreateTorus()->IsEdgeManifold(false));

    geometry::TriangleMesh mesh0;
    mesh0.vertices_ = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 0, 2}, {1, 0.5, 1}};
    mesh0.triangles_ = {{0, 1, 2}, {1, 2, 3}, {1, 2, 4}};
    EXPECT_FALSE(mesh0.IsEdgeManifold(true));
    EXPECT_FALSE(mesh0.IsEdgeManifold(false));

    geometry::TriangleMesh mesh1;
    mesh1.vertices_ = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 0, 2}};
    mesh1.triangles_ = {{0, 1, 2}, {1, 2, 3}};
    EXPECT_TRUE(mesh1.IsEdgeManifold(true));
    EXPECT_FALSE(mesh1.IsEdgeManifold(false));
}

TEST(TriangleMesh, IsVertexManifold) {
    EXPECT_TRUE(geometry::TriangleMesh::CreateBox()->IsVertexManifold());
    EXPECT_TRUE(geometry::TriangleMesh::CreateSphere()->IsVertexManifold());
    EXPECT_TRUE(geometry::TriangleMesh::CreateCylinder()->IsVertexManifold());
    EXPECT_TRUE(geometry::TriangleMesh::CreateCone()->IsVertexManifold());
    EXPECT_TRUE(geometry::TriangleMesh::CreateTorus()->IsVertexManifold());

    geometry::TriangleMesh mesh0;
    mesh0.vertices_ = {{0, 0, 0}, {1, 1, 1},  {1, 0, 1},
                       {0, 1, 1}, {1, 1, -1}, {1, 0, -1}};
    mesh0.triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 4, 5}};
    EXPECT_FALSE(mesh0.IsVertexManifold());

    geometry::TriangleMesh mesh1;
    mesh1.vertices_ = {{0, 0, 0},  {1, 1, 1},  {1, 0, 1}, {0, 1, 1},
                       {1, 1, -1}, {1, 0, -1}, {0, 1, -1}};
    mesh1.triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 4, 5}, {0, 5, 6}};
    EXPECT_FALSE(mesh1.IsVertexManifold());
}

TEST(TriangleMesh, IsSelfIntersecting) {
    EXPECT_FALSE(geometry::TriangleMesh::CreateBox()->IsSelfIntersecting());
    EXPECT_FALSE(geometry::TriangleMesh::CreateSphere()->IsSelfIntersecting());
    EXPECT_FALSE(
            geometry::TriangleMesh::CreateCylinder()->IsSelfIntersecting());
    EXPECT_FALSE(geometry::TriangleMesh::CreateCone()->IsSelfIntersecting());
    EXPECT_FALSE(geometry::TriangleMesh::CreateTorus()->IsSelfIntersecting());

    // simple intersection
    geometry::TriangleMesh mesh0;
    mesh0.vertices_ = {{0, 0, 0},      {0, 1, 0}, {1, 0, 0}, {1, 1, 0},
                       {0.5, 0.5, -1}, {0, 1, 1}, {1, 0, 1}};
    mesh0.triangles_ = {{0, 1, 2}, {1, 2, 3}, {4, 5, 6}};
    EXPECT_TRUE(mesh0.IsSelfIntersecting());

    // co-planar intersection
    geometry::TriangleMesh mesh1;
    mesh1.vertices_ = {{0, 0, 0},     {0, 1, 0},     {1, 0, 0},
                       {0.1, 0.1, 0}, {0.1, 1.1, 0}, {1.1, 0.1, 0}};
    mesh1.triangles_ = {{0, 1, 2}, {3, 4, 5}};
    EXPECT_TRUE(mesh1.IsSelfIntersecting());
}

TEST(TriangleMesh, GetVolume) {
    EXPECT_NEAR(geometry::TriangleMesh::CreateBox()->GetVolume(), 1.0, 0.01);
    EXPECT_NEAR(geometry::TriangleMesh::CreateSphere()->GetVolume(),
                4.0 / 3.0 * M_PI, 0.05);
}

TEST(TriangleMesh, ClusterConnectedTriangles) {
    // Test 1

    geometry::TriangleMesh mesh;
    mesh.vertices_ = {
            {0.000000, 0.000000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 1.000000}, {1.000000, 0.000000, 1.000000},
            {0.000000, 1.000000, 0.000000}, {1.000000, 1.000000, 0.000000},
            {0.000000, 1.000000, 1.000000}, {1.000000, 1.000000, 1.000000},
            {0.000000, 0.000000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 1.000000}, {1.000000, 0.000000, 1.000000},
            {0.000000, 1.000000, 0.000000}, {1.000000, 1.000000, 0.000000},
            {0.000000, 1.000000, 1.000000}, {1.000000, 1.000000, 1.000000},
            {0.000000, 0.000000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 1.000000}, {1.000000, 0.000000, 1.000000},
            {0.000000, 1.000000, 0.000000}, {1.000000, 1.000000, 0.000000},
            {0.000000, 1.000000, 1.000000}, {1.000000, 1.000000, 1.000000},
            {0.000000, 0.000000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 1.000000}, {1.000000, 0.000000, 1.000000},
            {0.000000, 1.000000, 0.000000}, {1.000000, 1.000000, 0.000000},
            {0.000000, 1.000000, 1.000000}, {1.000000, 1.000000, 1.000000}};
    mesh.triangles_ = {{4, 7, 5},    {4, 6, 7},    {0, 2, 4},    {2, 6, 4},
                       {0, 1, 2},    {1, 3, 2},    {1, 5, 7},    {1, 7, 3},
                       {2, 3, 7},    {2, 7, 6},    {0, 4, 1},    {1, 4, 5},
                       {12, 15, 13}, {12, 14, 15}, {8, 10, 12},  {10, 14, 12},
                       {8, 9, 10},   {9, 11, 10},  {9, 13, 15},  {9, 15, 11},
                       {10, 11, 15}, {10, 15, 14}, {8, 12, 9},   {9, 12, 13},
                       {20, 23, 21}, {20, 22, 23}, {16, 18, 20}, {18, 22, 20},
                       {16, 17, 18}, {17, 19, 18}, {17, 21, 23}, {17, 23, 19},
                       {18, 19, 23}, {18, 23, 22}, {16, 20, 17}, {17, 20, 21},
                       {28, 31, 29}, {28, 30, 31}, {24, 26, 28}, {26, 30, 28},
                       {24, 25, 26}, {25, 27, 26}, {25, 29, 31}, {25, 31, 27},
                       {26, 27, 31}, {26, 31, 30}, {24, 28, 25}, {25, 28, 29}};

    std::vector<int> gt_clusters = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    std::vector<size_t> gt_cluster_n_triangles = {12, 12, 12, 12};
    std::vector<double> gt_cluster_area = {6.0, 6.0, 6.0, 6.0};

    std::vector<int> clusters;
    std::vector<size_t> cluster_n_triangles;
    std::vector<double> cluster_area;
    std::tie(clusters, cluster_n_triangles, cluster_area) =
            mesh.ClusterConnectedTriangles();

    EXPECT_EQ(gt_clusters, clusters);
    EXPECT_EQ(cluster_n_triangles, gt_cluster_n_triangles);
    EXPECT_EQ(cluster_area, gt_cluster_area);

    // Test 2

    mesh.Clear();
    mesh.vertices_ = {
            {0.000000, 0.000000, 0.000000}, {0.500000, 0.866025, 0.000000},
            {1.000000, 0.000000, 0.000000}, {0.500000, 0.866025, 0.000000},

            {2.000000, 0.000000, 0.000000}, {2.500000, 0.866025, 0.000000},
            {3.000000, 0.000000, 0.000000}, {2.500000, 0.866025, 0.000000},

            {4.000000, 0.000000, 0.000000}, {4.500000, 0.866025, 0.000000},
            {5.000000, 0.000000, 0.000000}, {4.500000, 0.866025, 0.000000},

            {6.000000, 0.000000, 0.000000}, {6.500000, 0.866025, 0.000000},
            {7.000000, 0.000000, 0.000000}, {6.500000, 0.866025, 0.000000}};

    mesh.triangles_ = {{0, 1, 2},  {12, 14, 15}, {4, 5, 6},    {8, 10, 11},
                       {8, 9, 10}, {4, 6, 7},    {12, 13, 14}, {0, 2, 3}};

    gt_clusters = {0, 1, 2, 3, 3, 2, 1, 0};
    gt_cluster_n_triangles = {2, 2, 2, 2};
    gt_cluster_area = {0.866025, 0.866025, 0.866025, 0.866025};

    std::tie(clusters, cluster_n_triangles, cluster_area) =
            mesh.ClusterConnectedTriangles();

    EXPECT_EQ(gt_clusters, clusters);
    EXPECT_EQ(cluster_n_triangles, gt_cluster_n_triangles);
    EXPECT_EQ(cluster_area, gt_cluster_area);
}

TEST(TriangleMesh, RemoveTrianglesByMask) {
    geometry::TriangleMesh mesh_in;
    geometry::TriangleMesh mesh_gt;

    mesh_in.vertices_ = {
            {0.000000, 0.000000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 1.000000}, {1.000000, 0.000000, 1.000000},
            {0.000000, 1.000000, 0.000000}, {1.000000, 1.000000, 0.000000},
            {0.000000, 1.000000, 1.000000}, {1.000000, 1.000000, 1.000000},
            {0.488681, 0.819323, 0.638359}, {0.481610, 0.823406, 0.638359},
            {0.481610, 0.815241, 0.638359}, {0.752265, 0.943083, 0.892225},
            {0.745194, 0.947166, 0.892225}, {0.745194, 0.939001, 0.892225},
            {0.424683, 0.903004, 0.423322}, {0.417612, 0.907086, 0.423322},
            {0.417612, 0.898921, 0.423322}};
    mesh_in.vertex_normals_ = {
            {-0.577350, -0.577350, -0.577350}, {0.577350, -0.577350, -0.577350},
            {-0.577350, -0.577350, 0.577350},  {0.577350, -0.577350, 0.577350},
            {-0.577350, 0.577350, -0.577350},  {0.577350, 0.577350, -0.577350},
            {-0.577350, 0.577350, 0.577350},   {0.577350, 0.577350, 0.577350},
            {0.000000, 0.000000, 1.000000},    {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000},    {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000},    {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000},    {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000}};
    mesh_in.triangles_ = {{4, 7, 5},  {4, 6, 7},    {0, 2, 4},   {2, 6, 4},
                          {0, 1, 2},  {1, 3, 2},    {1, 5, 7},   {1, 7, 3},
                          {2, 3, 7},  {2, 7, 6},    {0, 4, 1},   {1, 4, 5},
                          {8, 9, 10}, {11, 12, 13}, {14, 15, 16}};
    mesh_in.triangle_normals_ = {
            {0.000000, 1.000000, 0.000000},  {0.000000, 1.000000, 0.000000},
            {-1.000000, 0.000000, 0.000000}, {-1.000000, 0.000000, 0.000000},
            {0.000000, -1.000000, 0.000000}, {0.000000, -1.000000, 0.000000},
            {1.000000, 0.000000, 0.000000},  {1.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 1.000000},  {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, -1.000000}, {0.000000, 0.000000, -1.000000},
            {0.000000, 0.000000, 1.000000},  {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000}};
    mesh_gt.vertices_ = {
            {0.000000, 0.000000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 1.000000}, {1.000000, 0.000000, 1.000000},
            {0.000000, 1.000000, 0.000000}, {1.000000, 1.000000, 0.000000},
            {0.000000, 1.000000, 1.000000}, {1.000000, 1.000000, 1.000000},
            {0.488681, 0.819323, 0.638359}, {0.481610, 0.823406, 0.638359},
            {0.481610, 0.815241, 0.638359}, {0.752265, 0.943083, 0.892225},
            {0.745194, 0.947166, 0.892225}, {0.745194, 0.939001, 0.892225},
            {0.424683, 0.903004, 0.423322}, {0.417612, 0.907086, 0.423322},
            {0.417612, 0.898921, 0.423322}};
    mesh_gt.vertex_normals_ = {
            {-0.577350, -0.577350, -0.577350}, {0.577350, -0.577350, -0.577350},
            {-0.577350, -0.577350, 0.577350},  {0.577350, -0.577350, 0.577350},
            {-0.577350, 0.577350, -0.577350},  {0.577350, 0.577350, -0.577350},
            {-0.577350, 0.577350, 0.577350},   {0.577350, 0.577350, 0.577350},
            {0.000000, 0.000000, 1.000000},    {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000},    {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000},    {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000},    {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000}};
    mesh_gt.triangles_ = {{4, 7, 5}, {4, 6, 7}, {0, 2, 4}, {2, 6, 4},
                          {0, 1, 2}, {1, 3, 2}, {1, 5, 7}, {1, 7, 3},
                          {2, 3, 7}, {2, 7, 6}, {0, 4, 1}, {1, 4, 5}};
    mesh_gt.triangle_normals_ = {
            {0.000000, 1.000000, 0.000000},  {0.000000, 1.000000, 0.000000},
            {-1.000000, 0.000000, 0.000000}, {-1.000000, 0.000000, 0.000000},
            {0.000000, -1.000000, 0.000000}, {0.000000, -1.000000, 0.000000},
            {1.000000, 0.000000, 0.000000},  {1.000000, 0.000000, 0.000000},
            {0.000000, 0.000000, 1.000000},  {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, -1.000000}, {0.000000, 0.000000, -1.000000}};
    std::vector<bool> triangles_to_remove = {false, false, false, false, false,
                                             false, false, false, false, false,
                                             false, false, true,  true,  true};

    mesh_in.RemoveTrianglesByMask(triangles_to_remove);

    ExpectMeshEQ(mesh_in, mesh_gt);
}

TEST(TriangleMesh, DeformAsRigidAsPossible) {
    geometry::TriangleMesh mesh_in;
    geometry::TriangleMesh mesh_gt;
    mesh_in.vertices_ = {
            {0.000000, 0.000000, 0.000000}, {0.000000, 1.000000, 0.000000},
            {1.000000, 1.000000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.500000, 0.500000, 0.000000}, {0.500000, 1.000000, 0.000000},
            {0.000000, 0.500000, 0.000000}, {0.500000, 0.000000, 0.000000},
            {1.000000, 0.500000, 0.000000}, {0.250000, 0.250000, 0.000000},
            {0.250000, 0.500000, 0.000000}, {0.000000, 0.250000, 0.000000},
            {0.750000, 0.750000, 0.000000}, {0.750000, 1.000000, 0.000000},
            {0.500000, 0.750000, 0.000000}, {0.250000, 1.000000, 0.000000},
            {0.000000, 0.750000, 0.000000}, {0.250000, 0.750000, 0.000000},
            {0.750000, 0.500000, 0.000000}, {1.000000, 0.750000, 0.000000},
            {0.250000, 0.000000, 0.000000}, {0.500000, 0.250000, 0.000000},
            {0.750000, 0.000000, 0.000000}, {1.000000, 0.250000, 0.000000},
            {0.750000, 0.250000, 0.000000}, {0.125000, 0.125000, 0.000000},
            {0.125000, 0.250000, 0.000000}, {0.000000, 0.125000, 0.000000},
            {0.375000, 0.375000, 0.000000}, {0.375000, 0.500000, 0.000000},
            {0.250000, 0.375000, 0.000000}, {0.125000, 0.500000, 0.000000},
            {0.000000, 0.375000, 0.000000}, {0.125000, 0.375000, 0.000000},
            {0.625000, 0.625000, 0.000000}, {0.625000, 0.750000, 0.000000},
            {0.500000, 0.625000, 0.000000}, {0.875000, 0.875000, 0.000000},
            {0.875000, 1.000000, 0.000000}, {0.750000, 0.875000, 0.000000},
            {0.625000, 1.000000, 0.000000}, {0.500000, 0.875000, 0.000000},
            {0.625000, 0.875000, 0.000000}, {0.375000, 1.000000, 0.000000},
            {0.250000, 0.875000, 0.000000}, {0.375000, 0.875000, 0.000000},
            {0.125000, 1.000000, 0.000000}, {0.000000, 0.875000, 0.000000},
            {0.125000, 0.875000, 0.000000}, {0.000000, 0.625000, 0.000000},
            {0.125000, 0.625000, 0.000000}, {0.125000, 0.750000, 0.000000},
            {0.375000, 0.625000, 0.000000}, {0.375000, 0.750000, 0.000000},
            {0.250000, 0.625000, 0.000000}, {0.875000, 0.750000, 0.000000},
            {1.000000, 0.875000, 0.000000}, {0.625000, 0.500000, 0.000000},
            {0.750000, 0.625000, 0.000000}, {0.875000, 0.500000, 0.000000},
            {1.000000, 0.625000, 0.000000}, {0.875000, 0.625000, 0.000000},
            {0.375000, 0.250000, 0.000000}, {0.500000, 0.375000, 0.000000},
            {0.125000, 0.000000, 0.000000}, {0.250000, 0.125000, 0.000000},
            {0.375000, 0.000000, 0.000000}, {0.500000, 0.125000, 0.000000},
            {0.375000, 0.125000, 0.000000}, {0.625000, 0.000000, 0.000000},
            {0.750000, 0.125000, 0.000000}, {0.625000, 0.125000, 0.000000},
            {0.875000, 0.000000, 0.000000}, {1.000000, 0.125000, 0.000000},
            {0.875000, 0.125000, 0.000000}, {1.000000, 0.375000, 0.000000},
            {0.875000, 0.375000, 0.000000}, {0.875000, 0.250000, 0.000000},
            {0.625000, 0.375000, 0.000000}, {0.625000, 0.250000, 0.000000},
            {0.750000, 0.375000, 0.000000}};

    mesh_in.triangles_ = {
            {0, 25, 27},  {25, 9, 26},  {26, 11, 27}, {25, 26, 27},
            {9, 28, 30},  {28, 4, 29},  {29, 10, 30}, {28, 29, 30},
            {10, 31, 33}, {31, 6, 32},  {32, 11, 33}, {31, 32, 33},
            {9, 30, 26},  {30, 10, 33}, {33, 11, 26}, {30, 33, 26},
            {4, 34, 36},  {34, 12, 35}, {35, 14, 36}, {34, 35, 36},
            {12, 37, 39}, {37, 2, 38},  {38, 13, 39}, {37, 38, 39},
            {13, 40, 42}, {40, 5, 41},  {41, 14, 42}, {40, 41, 42},
            {12, 39, 35}, {39, 13, 42}, {42, 14, 35}, {39, 42, 35},
            {5, 43, 45},  {43, 15, 44}, {44, 17, 45}, {43, 44, 45},
            {15, 46, 48}, {46, 1, 47},  {47, 16, 48}, {46, 47, 48},
            {16, 49, 51}, {49, 6, 50},  {50, 17, 51}, {49, 50, 51},
            {15, 48, 44}, {48, 16, 51}, {51, 17, 44}, {48, 51, 44},
            {4, 36, 29},  {36, 14, 52}, {52, 10, 29}, {36, 52, 29},
            {14, 41, 53}, {41, 5, 45},  {45, 17, 53}, {41, 45, 53},
            {17, 50, 54}, {50, 6, 31},  {31, 10, 54}, {50, 31, 54},
            {14, 53, 52}, {53, 17, 54}, {54, 10, 52}, {53, 54, 52},
            {2, 37, 56},  {37, 12, 55}, {55, 19, 56}, {37, 55, 56},
            {12, 34, 58}, {34, 4, 57},  {57, 18, 58}, {34, 57, 58},
            {18, 59, 61}, {59, 8, 60},  {60, 19, 61}, {59, 60, 61},
            {12, 58, 55}, {58, 18, 61}, {61, 19, 55}, {58, 61, 55},
            {4, 28, 63},  {28, 9, 62},  {62, 21, 63}, {28, 62, 63},
            {9, 25, 65},  {25, 0, 64},  {64, 20, 65}, {25, 64, 65},
            {20, 66, 68}, {66, 7, 67},  {67, 21, 68}, {66, 67, 68},
            {9, 65, 62},  {65, 20, 68}, {68, 21, 62}, {65, 68, 62},
            {7, 69, 71},  {69, 22, 70}, {70, 24, 71}, {69, 70, 71},
            {22, 72, 74}, {72, 3, 73},  {73, 23, 74}, {72, 73, 74},
            {23, 75, 77}, {75, 8, 76},  {76, 24, 77}, {75, 76, 77},
            {22, 74, 70}, {74, 23, 77}, {77, 24, 70}, {74, 77, 70},
            {4, 63, 57},  {63, 21, 78}, {78, 18, 57}, {63, 78, 57},
            {21, 67, 79}, {67, 7, 71},  {71, 24, 79}, {67, 71, 79},
            {24, 76, 80}, {76, 8, 59},  {59, 18, 80}, {76, 59, 80},
            {21, 79, 78}, {79, 24, 80}, {80, 18, 78}, {79, 80, 78}};
    mesh_gt.vertices_ = {
            {0.000000, 0.000000, 0.000000}, {0.000000, 1.000000, 0.000000},
            {1.000000, 1.000000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.500000, 0.500000, 0.400000}, {0.500000, 1.000000, 0.000000},
            {0.000000, 0.500000, 0.000000}, {0.500000, 0.000000, 0.000000},
            {1.000000, 0.500000, 0.000000}, {0.250492, 0.250492, 0.040018},
            {0.248198, 0.500000, 0.102907}, {0.000000, 0.250000, 0.000000},
            {0.749508, 0.749508, 0.040018}, {0.750000, 1.000000, 0.000000},
            {0.500000, 0.751802, 0.102907}, {0.250000, 1.000000, 0.000000},
            {0.000000, 0.750000, 0.000000}, {0.250492, 0.749508, 0.040018},
            {0.751802, 0.500000, 0.102907}, {1.000000, 0.750000, 0.000000},
            {0.250000, 0.000000, 0.000000}, {0.500000, 0.248198, 0.102907},
            {0.750000, 0.000000, 0.000000}, {1.000000, 0.250000, 0.000000},
            {0.749508, 0.250492, 0.040018}, {0.125000, 0.125000, 0.000000},
            {0.125000, 0.250000, 0.000000}, {0.000000, 0.125000, 0.000000},
            {0.366700, 0.366700, 0.181470}, {0.361404, 0.500000, 0.242836},
            {0.248165, 0.374408, 0.083221}, {0.125000, 0.500000, 0.000000},
            {0.000000, 0.375000, 0.000000}, {0.125000, 0.375000, 0.000000},
            {0.633300, 0.633300, 0.181470}, {0.625592, 0.751835, 0.083221},
            {0.500000, 0.638596, 0.242836}, {0.875000, 0.875000, 0.000000},
            {0.875000, 1.000000, 0.000000}, {0.750000, 0.875000, 0.000000},
            {0.625000, 1.000000, 0.000000}, {0.500000, 0.875000, 0.000000},
            {0.625000, 0.875000, 0.000000}, {0.375000, 1.000000, 0.000000},
            {0.250000, 0.875000, 0.000000}, {0.375000, 0.875000, 0.000000},
            {0.125000, 1.000000, 0.000000}, {0.000000, 0.875000, 0.000000},
            {0.125000, 0.875000, 0.000000}, {0.000000, 0.625000, 0.000000},
            {0.125000, 0.625000, 0.000000}, {0.125000, 0.750000, 0.000000},
            {0.366700, 0.633300, 0.181470}, {0.374408, 0.751835, 0.083221},
            {0.248165, 0.625592, 0.083221}, {0.875000, 0.750000, 0.000000},
            {1.000000, 0.875000, 0.000000}, {0.638596, 0.500000, 0.242836},
            {0.751835, 0.625592, 0.083221}, {0.875000, 0.500000, 0.000000},
            {1.000000, 0.625000, 0.000000}, {0.875000, 0.625000, 0.000000},
            {0.374408, 0.248165, 0.083221}, {0.500000, 0.361404, 0.242836},
            {0.125000, 0.000000, 0.000000}, {0.250000, 0.125000, 0.000000},
            {0.375000, 0.000000, 0.000000}, {0.500000, 0.125000, 0.000000},
            {0.375000, 0.125000, 0.000000}, {0.625000, 0.000000, 0.000000},
            {0.750000, 0.125000, 0.000000}, {0.625000, 0.125000, 0.000000},
            {0.875000, 0.000000, 0.000000}, {1.000000, 0.125000, 0.000000},
            {0.875000, 0.125000, 0.000000}, {1.000000, 0.375000, 0.000000},
            {0.875000, 0.375000, 0.000000}, {0.875000, 0.250000, 0.000000},
            {0.633300, 0.366700, 0.181470}, {0.625592, 0.248165, 0.083221},
            {0.751835, 0.374408, 0.083221}};

    mesh_gt.triangles_ = {
            {0, 25, 27},  {25, 9, 26},  {26, 11, 27}, {25, 26, 27},
            {9, 28, 30},  {28, 4, 29},  {29, 10, 30}, {28, 29, 30},
            {10, 31, 33}, {31, 6, 32},  {32, 11, 33}, {31, 32, 33},
            {9, 30, 26},  {30, 10, 33}, {33, 11, 26}, {30, 33, 26},
            {4, 34, 36},  {34, 12, 35}, {35, 14, 36}, {34, 35, 36},
            {12, 37, 39}, {37, 2, 38},  {38, 13, 39}, {37, 38, 39},
            {13, 40, 42}, {40, 5, 41},  {41, 14, 42}, {40, 41, 42},
            {12, 39, 35}, {39, 13, 42}, {42, 14, 35}, {39, 42, 35},
            {5, 43, 45},  {43, 15, 44}, {44, 17, 45}, {43, 44, 45},
            {15, 46, 48}, {46, 1, 47},  {47, 16, 48}, {46, 47, 48},
            {16, 49, 51}, {49, 6, 50},  {50, 17, 51}, {49, 50, 51},
            {15, 48, 44}, {48, 16, 51}, {51, 17, 44}, {48, 51, 44},
            {4, 36, 29},  {36, 14, 52}, {52, 10, 29}, {36, 52, 29},
            {14, 41, 53}, {41, 5, 45},  {45, 17, 53}, {41, 45, 53},
            {17, 50, 54}, {50, 6, 31},  {31, 10, 54}, {50, 31, 54},
            {14, 53, 52}, {53, 17, 54}, {54, 10, 52}, {53, 54, 52},
            {2, 37, 56},  {37, 12, 55}, {55, 19, 56}, {37, 55, 56},
            {12, 34, 58}, {34, 4, 57},  {57, 18, 58}, {34, 57, 58},
            {18, 59, 61}, {59, 8, 60},  {60, 19, 61}, {59, 60, 61},
            {12, 58, 55}, {58, 18, 61}, {61, 19, 55}, {58, 61, 55},
            {4, 28, 63},  {28, 9, 62},  {62, 21, 63}, {28, 62, 63},
            {9, 25, 65},  {25, 0, 64},  {64, 20, 65}, {25, 64, 65},
            {20, 66, 68}, {66, 7, 67},  {67, 21, 68}, {66, 67, 68},
            {9, 65, 62},  {65, 20, 68}, {68, 21, 62}, {65, 68, 62},
            {7, 69, 71},  {69, 22, 70}, {70, 24, 71}, {69, 70, 71},
            {22, 72, 74}, {72, 3, 73},  {73, 23, 74}, {72, 73, 74},
            {23, 75, 77}, {75, 8, 76},  {76, 24, 77}, {75, 76, 77},
            {22, 74, 70}, {74, 23, 77}, {77, 24, 70}, {74, 77, 70},
            {4, 63, 57},  {63, 21, 78}, {78, 18, 57}, {63, 78, 57},
            {21, 67, 79}, {67, 7, 71},  {71, 24, 79}, {67, 71, 79},
            {24, 76, 80}, {76, 8, 59},  {59, 18, 80}, {76, 59, 80},
            {21, 79, 78}, {79, 24, 80}, {80, 18, 78}, {79, 80, 78}};

    std::vector<int> constraint_ids = {
            1,  46, 47, 48, 16, 51, 49, 50, 6,  31, 33, 32, 11, 26, 27,
            25, 0,  64, 65, 20, 66, 68, 67, 7,  69, 71, 70, 22, 72, 74,
            73, 3,  15, 44, 43, 45, 5,  41, 40, 42, 13, 39, 37, 38, 2,
            56, 55, 19, 61, 60, 59, 8,  76, 75, 77, 23, 4};
    std::vector<Eigen::Vector3d> constraint_pos = {
            {0.000000, 1.000000, 0.000000}, {0.125000, 1.000000, 0.000000},
            {0.000000, 0.875000, 0.000000}, {0.125000, 0.875000, 0.000000},
            {0.000000, 0.750000, 0.000000}, {0.125000, 0.750000, 0.000000},
            {0.000000, 0.625000, 0.000000}, {0.125000, 0.625000, 0.000000},
            {0.000000, 0.500000, 0.000000}, {0.125000, 0.500000, 0.000000},
            {0.125000, 0.375000, 0.000000}, {0.000000, 0.375000, 0.000000},
            {0.000000, 0.250000, 0.000000}, {0.125000, 0.250000, 0.000000},
            {0.000000, 0.125000, 0.000000}, {0.125000, 0.125000, 0.000000},
            {0.000000, 0.000000, 0.000000}, {0.125000, 0.000000, 0.000000},
            {0.250000, 0.125000, 0.000000}, {0.250000, 0.000000, 0.000000},
            {0.375000, 0.000000, 0.000000}, {0.375000, 0.125000, 0.000000},
            {0.500000, 0.125000, 0.000000}, {0.500000, 0.000000, 0.000000},
            {0.625000, 0.000000, 0.000000}, {0.625000, 0.125000, 0.000000},
            {0.750000, 0.125000, 0.000000}, {0.750000, 0.000000, 0.000000},
            {0.875000, 0.000000, 0.000000}, {0.875000, 0.125000, 0.000000},
            {1.000000, 0.125000, 0.000000}, {1.000000, 0.000000, 0.000000},
            {0.250000, 1.000000, 0.000000}, {0.250000, 0.875000, 0.000000},
            {0.375000, 1.000000, 0.000000}, {0.375000, 0.875000, 0.000000},
            {0.500000, 1.000000, 0.000000}, {0.500000, 0.875000, 0.000000},
            {0.625000, 1.000000, 0.000000}, {0.625000, 0.875000, 0.000000},
            {0.750000, 1.000000, 0.000000}, {0.750000, 0.875000, 0.000000},
            {0.875000, 0.875000, 0.000000}, {0.875000, 1.000000, 0.000000},
            {1.000000, 1.000000, 0.000000}, {1.000000, 0.875000, 0.000000},
            {0.875000, 0.750000, 0.000000}, {1.000000, 0.750000, 0.000000},
            {0.875000, 0.625000, 0.000000}, {1.000000, 0.625000, 0.000000},
            {0.875000, 0.500000, 0.000000}, {1.000000, 0.500000, 0.000000},
            {0.875000, 0.375000, 0.000000}, {1.000000, 0.375000, 0.000000},
            {0.875000, 0.250000, 0.000000}, {1.000000, 0.250000, 0.000000},
            {0.500000, 0.500000, 0.400000}};

    auto mesh_deform =
            mesh_in.DeformAsRigidAsPossible(constraint_ids, constraint_pos, 50);
    ExpectMeshEQ(*mesh_deform, mesh_gt, 1e-5);
}

TEST(TriangleMesh, SelectByIndex) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {360.784314, 717.647059, 800.000000},
            {90.196078, 133.333333, 517.647059},
            {913.725490, 858.823529, 200.000000},
            {647.058824, 360.784314, 286.274510},
            {811.764706, 752.941176, 352.941176},
            {764.705882, 333.333333, 533.333333},
            {529.411765, 454.901961, 639.215686},
            {39.215686, 411.764706, 694.117647},
            {600.000000, 129.411765, 78.431373},
            {447.058824, 486.274510, 792.156863},
            {588.235294, 576.470588, 529.411765},
            {47.058824, 525.490196, 176.470588},
            {39.215686, 435.294118, 929.411765},
            {866.666667, 545.098039, 737.254902},
            {945.098039, 98.039216, 274.509804},
            {341.176471, 47.058824, 764.705882},
            {831.372549, 713.725490, 631.372549},
            {811.764706, 682.352941, 909.803922},
            {290.196078, 356.862745, 874.509804},
            {90.196078, 960.784314, 866.666667},
            {349.019608, 803.921569, 917.647059},
            {886.274510, 945.098039, 54.901961},
            {439.215686, 117.647059, 588.235294},
            {635.294118, 984.313725, 337.254902},
            {258.823529, 647.058824, 549.019608}};

    std::vector<Eigen::Vector3d> ref_vertex_normals = {
            {717.647059, 800.000000, 674.509804},
            {133.333333, 517.647059, 74.509804},
            {858.823529, 200.000000, 792.156863},
            {360.784314, 286.274510, 329.411765},
            {752.941176, 352.941176, 996.078431},
            {333.333333, 533.333333, 215.686275},
            {454.901961, 639.215686, 713.725490},
            {411.764706, 694.117647, 670.588235},
            {129.411765, 78.431373, 376.470588},
            {486.274510, 792.156863, 635.294118},
            {576.470588, 529.411765, 592.156863},
            {525.490196, 176.470588, 925.490196},
            {435.294118, 929.411765, 929.411765},
            {545.098039, 737.254902, 929.411765},
            {98.039216, 274.509804, 239.215686},
            {47.058824, 764.705882, 800.000000},
            {713.725490, 631.372549, 513.725490},
            {682.352941, 909.803922, 482.352941},
            {356.862745, 874.509804, 341.176471},
            {960.784314, 866.666667, 164.705882},
            {803.921569, 917.647059, 66.666667},
            {945.098039, 54.901961, 921.568627},
            {117.647059, 588.235294, 576.470588},
            {984.313725, 337.254902, 894.117647},
            {647.058824, 549.019608, 917.647059}};

    std::vector<Eigen::Vector3d> ref_vertex_colors = {
            {396.078431, 392.156863, 596.078431},
            {160.784314, 862.745098, 462.745098},
            {882.352941, 180.392157, 776.470588},
            {200.000000, 698.039216, 886.274510},
            {286.274510, 90.196078, 933.333333},
            {756.862745, 133.333333, 74.509804},
            {43.137255, 772.549020, 858.823529},
            {882.352941, 156.862745, 827.450980},
            {239.215686, 341.176471, 921.568627},
            {635.294118, 184.313725, 623.529412},
            {274.509804, 807.843137, 745.098039},
            {47.058824, 800.000000, 678.431373},
            {172.549020, 796.078431, 654.901961},
            {513.725490, 341.176471, 94.117647},
            {764.705882, 533.333333, 474.509804},
            {337.254902, 133.333333, 3.921569},
            {266.666667, 329.411765, 756.862745},
            {831.372549, 870.588235, 976.470588},
            {125.490196, 82.352941, 784.313725},
            {450.980392, 411.764706, 403.921569},
            {654.901961, 439.215686, 396.078431},
            {168.627451, 525.490196, 596.078431},
            {94.117647, 945.098039, 274.509804},
            {203.921569, 400.000000, 678.431373},
            {141.176471, 560.784314, 870.588235}};

    std::vector<Eigen::Vector3i> ref_triangles = {
            {3, 6, 9},    {12, 15, 18}, {8, 11, 14},  {17, 20, 23}, {4, 7, 10},
            {13, 16, 19}, {0, 3, 6},    {9, 12, 15},  {18, 21, 24}, {5, 8, 11},
            {14, 17, 20}, {1, 4, 7},    {10, 13, 16}, {24, 12, 10}, {6, 9, 12},
            {15, 18, 21}, {2, 5, 8},    {11, 14, 17}, {9, 10, 15},  {7, 10, 13},
            {16, 19, 22}};

    std::vector<Eigen::Vector3d> ref_triangle_normals = {
            {909.803922, 274.509804, 364.705882},
            {635.294118, 15.686275, 152.941176},
            {262.745098, 156.862745, 207.843137},
            {392.156863, 635.294118, 133.333333},
            {35.294118, 698.039216, 341.176471},
            {698.039216, 749.019608, 588.235294},
            {839.215686, 909.803922, 274.509804},
            {364.705882, 635.294118, 15.686275},
            {152.941176, 996.078431, 611.764706},
            {90.196078, 262.745098, 156.862745},
            {207.843137, 392.156863, 635.294118},
            {811.764706, 35.294118, 698.039216},
            {341.176471, 698.039216, 749.019608},
            {933.333333, 796.078431, 992.156863},
            {274.509804, 364.705882, 635.294118},
            {15.686275, 152.941176, 996.078431},
            {647.058824, 90.196078, 262.745098},
            {156.862745, 207.843137, 392.156863},
            {682.352941, 294.117647, 149.019608},
            {698.039216, 341.176471, 698.039216},
            {749.019608, 588.235294, 243.137255}};

    int size = 1000;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.vertex_colors_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    Rand(tm.vertices_, dmin, dmax, 0);
    Rand(tm.vertex_normals_, dmin, dmax, 1);
    Rand(tm.vertex_colors_, dmin, dmax, 2);
    Rand(tm.triangles_, imin, imax, 3);
    Rand(tm.triangle_normals_, dmin, dmax, 4);

    std::vector<size_t> indices(size / 40);
    Rand(indices, 0, size - 1, 0);

    auto output_tm = tm.SelectByIndex(indices);

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_vertex_normals, output_tm->vertex_normals_);
    ExpectEQ(ref_vertex_colors, output_tm->vertex_colors_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
    ExpectEQ(ref_triangle_normals, output_tm->triangle_normals_);
}

TEST(TriangleMesh, CropTriangleMesh) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {615.686275, 639.215686, 517.647059},
            {615.686275, 760.784314, 772.549020},
            {678.431373, 643.137255, 443.137255},
            {345.098039, 529.411765, 454.901961},
            {360.784314, 576.470588, 592.156863},
            {317.647059, 666.666667, 525.490196}};

    std::vector<Eigen::Vector3d> ref_vertex_normals = {
            {639.215686, 517.647059, 400.000000},
            {760.784314, 772.549020, 282.352941},
            {643.137255, 443.137255, 266.666667},
            {529.411765, 454.901961, 639.215686},
            {576.470588, 592.156863, 662.745098},
            {666.666667, 525.490196, 847.058824}};

    std::vector<Eigen::Vector3d> ref_vertex_colors = {
            {647.058824, 325.490196, 603.921569},
            {941.176471, 121.568627, 513.725490},
            {807.843137, 309.803922, 3.921569},
            {686.274510, 43.137255, 772.549020},
            {576.470588, 698.039216, 831.372549},
            {854.901961, 341.176471, 878.431373}};

    std::vector<Eigen::Vector3i> ref_triangles = {{1, 0, 3}, {5, 2, 4}};

    std::vector<Eigen::Vector3d> ref_triangle_normals = {
            {125.490196, 160.784314, 913.725490},
            {764.705882, 901.960784, 807.843137}};

    int size = 1000;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.vertex_colors_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    Rand(tm.vertices_, dmin, dmax, 0);
    Rand(tm.vertex_normals_, dmin, dmax, 1);
    Rand(tm.vertex_colors_, dmin, dmax, 2);
    Rand(tm.triangles_, imin, imax, 3);
    Rand(tm.triangle_normals_, dmin, dmax, 4);

    Eigen::Vector3d cropBoundMin(300.0, 300.0, 300.0);
    Eigen::Vector3d cropBoundMax(800.0, 800.0, 800.0);

    auto output_tm = tm.Crop(
            geometry::AxisAlignedBoundingBox(cropBoundMin, cropBoundMax));

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_vertex_normals, output_tm->vertex_normals_);
    ExpectEQ(ref_vertex_colors, output_tm->vertex_colors_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
    ExpectEQ(ref_triangle_normals, output_tm->triangle_normals_);
}

TEST(TriangleMesh, CreateFromPointCloudPoisson) {
    geometry::PointCloud pcd;
    pcd.points_ = {
            {-0.215279, 0.121252, 0.965784},  {0.079266, 0.643799, 0.755848},
            {0.001534, -0.691225, 0.720568},  {0.663793, 0.567055, 0.478929},
            {-0.929397, -0.262081, 0.238929}, {0.741441, -0.628480, 0.209423},
            {-0.844085, 0.510828, 0.131410},  {0.001478, -0.999346, 0.006834},
            {-0.075616, 0.987936, -0.077669}, {-0.599516, -0.738522, -0.297292},
            {0.952159, -0.061734, -0.284187}, {0.616751, -0.543117, -0.562931},
            {-0.066283, 0.653032, -0.748836}, {0.408743, 0.133839, -0.900872},
            {-0.165481, -0.187750, -0.964982}};
    pcd.normals_ = {
            {-0.227789, 0.134744, 0.961608},  {0.084291, 0.646929, 0.752716},
            {-0.002898, -0.694359, 0.717585}, {0.666542, 0.565546, 0.476952},
            {-0.930073, -0.260750, 0.237944}, {0.740792, -0.629608, 0.208559},
            {-0.843751, 0.511569, 0.130869},  {0.001436, -0.999352, 0.006806},
            {-0.076095, 0.987987, -0.077349}, {-0.598343, -0.739977, -0.296067},
            {0.952641, -0.060018, -0.283016}, {0.620242, -0.541600, -0.560605},
            {-0.071145, 0.656184, -0.745733}, {0.414599, 0.141599, -0.897109},
            {-0.172774, -0.204293, -0.960815}};
    pcd.colors_ = {
            {0.380208, 0.759545, 0.171340}, {0.288485, 0.063605, 0.643738},
            {0.997344, 0.888908, 0.802631}, {0.249312, 0.621767, 0.718174},
            {0.953116, 0.032469, 0.828249}, {0.763606, 0.914161, 0.458050},
            {0.735258, 0.974643, 0.458129}, {0.985589, 0.649781, 0.064284},
            {0.714224, 0.413067, 0.800399}, {0.433070, 0.962528, 0.138826},
            {0.066043, 0.867413, 0.276809}, {0.321559, 0.662692, 0.011849},
            {0.353338, 0.784374, 0.899556}, {0.248052, 0.431480, 0.339511},
            {0.880225, 0.614243, 0.379607}};
    geometry::TriangleMesh mesh_gt;
    mesh_gt.vertices_ = {{-0.535122, -0.678988, -0.546102},
                         {-0.856971, -0.552207, -0.546102},
                         {0.011381, -0.852840, -0.546102},
                         {-1.081624, -0.478258, -0.546102},
                         {0.557883, -0.571511, -0.546102},
                         {0.576765, -0.552207, -0.546102},
                         {0.739454, -0.005705, -0.546102},
                         {-1.081624, 0.339532, -0.546102},
                         {-0.720499, 0.540798, -0.546102},
                         {-0.535122, 0.637316, -0.546102},
                         {0.011381, 0.788968, -0.546102},
                         {0.559266, 0.540798, -0.546102},
                         {0.557883, 0.542438, -0.546102},
                         {-0.535122, -0.552207, -0.761037},
                         {0.011381, -0.552207, -0.838612},
                         {-1.081624, -0.005705, -1.024482},
                         {-0.535122, -0.005705, -1.038080},
                         {0.011381, -0.005705, -1.001911},
                         {0.557883, -0.552207, -0.565591},
                         {0.557883, -0.005705, -0.755925},
                         {-0.535122, 0.540798, -0.678056},
                         {0.011381, 0.540798, -0.807635},
                         {0.557883, 0.540798, -0.547422},
                         {-0.535122, -0.868303, 0.000401},
                         {-0.957409, -0.552207, 0.000401},
                         {0.011381, -0.981478, 0.000401},
                         {-1.081624, -0.413615, 0.000401},
                         {0.557883, -0.735421, 0.000401},
                         {0.756038, -0.552207, 0.000401},
                         {1.045917, -0.005705, 0.000401},
                         {-1.081624, 0.279246, 0.000401},
                         {-0.856915, 0.540798, 0.000401},
                         {-0.535122, 0.806682, 0.000401},
                         {0.011381, 0.995774, 0.000401},
                         {0.876659, 0.540798, 0.000401},
                         {0.557883, 0.851917, 0.000401},
                         {0.011381, -0.775497, 0.546903},
                         {-0.494233, -0.552207, 0.546903},
                         {-0.535122, -0.513868, 0.546903},
                         {-0.768247, -0.005705, 0.546903},
                         {0.557883, -0.601114, 0.546903},
                         {0.621286, -0.552207, 0.546903},
                         {0.891579, -0.005705, 0.546903},
                         {-0.535122, 0.523186, 0.546903},
                         {-0.516131, 0.540798, 0.546903},
                         {0.011381, 0.784610, 0.546903},
                         {0.613233, 0.540798, 0.546903},
                         {0.557883, 0.585415, 0.546903},
                         {-0.535122, -0.552207, 0.520166},
                         {-1.081624, -0.005705, 0.145186},
                         {-0.535122, 0.540798, 0.529053},
                         {0.011381, -0.552207, 0.785752},
                         {-0.535122, -0.005705, 0.773541},
                         {0.011381, -0.005705, 1.045411},
                         {0.557883, -0.552207, 0.612610},
                         {0.557883, -0.005705, 0.880973},
                         {0.011381, 0.540798, 0.802148},
                         {0.557883, 0.540798, 0.593895}};
    mesh_gt.vertex_normals_ = {{-0.363328, -0.449218, -0.596538},
                               {-0.363328, -0.449218, -0.596538},
                               {-0.363328, -0.449218, -0.596538},
                               {-0.363328, -0.449218, -0.596538},
                               {0.748097, -0.287048, -0.401288},
                               {0.748097, -0.287048, -0.401288},
                               {0.624682, -0.539519, -0.560146},
                               {-0.067839, 0.778945, -0.391656},
                               {-0.067839, 0.778945, -0.391656},
                               {-0.067839, 0.778945, -0.391656},
                               {-0.071381, 0.660859, -0.743014},
                               {0.375628, 0.123591, -0.806610},
                               {0.375628, 0.123591, -0.806610},
                               {-0.363328, -0.449218, -0.596538},
                               {-0.363328, -0.449218, -0.596538},
                               {-0.363328, -0.449218, -0.596538},
                               {-0.363328, -0.449218, -0.596538},
                               {-0.176322, -0.208808, -0.957378},
                               {0.748097, -0.287048, -0.401288},
                               {0.748097, -0.287048, -0.401288},
                               {-0.067839, 0.778945, -0.391656},
                               {-0.067839, 0.778945, -0.391656},
                               {0.414932, 0.141672, -0.897776},
                               {-0.596520, -0.737720, -0.301946},
                               {-0.596520, -0.737720, -0.301946},
                               {-0.363328, -0.449218, -0.596538},
                               {-0.363328, -0.449218, -0.596538},
                               {0.748097, -0.287048, -0.401288},
                               {0.748097, -0.287048, -0.401288},
                               {0.953452, -0.063973, -0.286170},
                               {-0.067839, 0.778945, -0.391656},
                               {-0.067839, 0.778945, -0.391656},
                               {-0.067839, 0.778945, -0.391656},
                               {-0.076419, 0.990524, -0.082928},
                               {0.375628, 0.123591, -0.806610},
                               {0.375628, 0.123591, -0.806610},
                               {-0.005705, -0.991160, 0.013986},
                               {-0.005705, -0.991160, 0.013986},
                               {-0.919808, -0.270837, 0.240837},
                               {-0.919808, -0.270837, 0.240837},
                               {0.670787, -0.570145, 0.185455},
                               {0.744380, -0.632658, 0.209540},
                               {0.670787, -0.570145, 0.185455},
                               {-0.842126, 0.510549, 0.137740},
                               {-0.506191, 0.304790, 0.516282},
                               {-0.506191, 0.304790, 0.516282},
                               {0.358188, 0.573235, 0.581938},
                               {0.358188, 0.573235, 0.581938},
                               {-0.298994, -0.629163, 0.308635},
                               {-0.919808, -0.270837, 0.240837},
                               {-0.842126, 0.510549, 0.137740},
                               {-0.009941, -0.693804, 0.708891},
                               {-0.298994, -0.629163, 0.308635},
                               {-0.298994, -0.629163, 0.308635},
                               {0.670787, -0.570145, 0.185455},
                               {0.670787, -0.570145, 0.185455},
                               {-0.232912, 0.137861, 0.956863},
                               {0.358188, 0.573235, 0.581938}};
    mesh_gt.vertex_colors_ = {
            {0.651185, 0.780322, 0.270666}, {0.651185, 0.780322, 0.270666},
            {0.651185, 0.780322, 0.270666}, {0.651185, 0.780322, 0.270666},
            {0.213958, 0.758281, 0.162138}, {0.213958, 0.758281, 0.162138},
            {0.319808, 0.664247, 0.014295}, {0.535119, 0.601156, 0.828728},
            {0.535119, 0.601156, 0.828728}, {0.535119, 0.601156, 0.828728},
            {0.356296, 0.781393, 0.898403}, {0.280560, 0.453637, 0.352787},
            {0.280560, 0.453637, 0.352787}, {0.651185, 0.780322, 0.270666},
            {0.651185, 0.780322, 0.270666}, {0.651185, 0.780322, 0.270666},
            {0.651185, 0.780322, 0.270666}, {0.876498, 0.616945, 0.377835},
            {0.213958, 0.758281, 0.162138}, {0.213958, 0.758281, 0.162138},
            {0.535119, 0.601156, 0.828728}, {0.535119, 0.601156, 0.828728},
            {0.248333, 0.431672, 0.339626}, {0.436619, 0.959563, 0.140971},
            {0.436619, 0.959563, 0.140971}, {0.651185, 0.780322, 0.270666},
            {0.651185, 0.780322, 0.270666}, {0.213958, 0.758281, 0.162138},
            {0.213958, 0.758281, 0.162138}, {0.068450, 0.865637, 0.274943},
            {0.535119, 0.601156, 0.828728}, {0.535119, 0.601156, 0.828728},
            {0.535119, 0.601156, 0.828728}, {0.711309, 0.416128, 0.800860},
            {0.280560, 0.453637, 0.352787}, {0.280560, 0.453637, 0.352787},
            {0.985049, 0.646890, 0.076101}, {0.985049, 0.646890, 0.076101},
            {0.953348, 0.044255, 0.821904}, {0.953348, 0.044255, 0.821904},
            {0.742035, 0.885688, 0.458892}, {0.763420, 0.913915, 0.458058},
            {0.742035, 0.885688, 0.458892}, {0.732370, 0.972691, 0.455932},
            {0.557746, 0.854674, 0.323112}, {0.557746, 0.854674, 0.323112},
            {0.284898, 0.359292, 0.669062}, {0.284898, 0.359292, 0.669062},
            {0.962866, 0.528193, 0.561334}, {0.953348, 0.044255, 0.821904},
            {0.732370, 0.972691, 0.455932}, {0.996524, 0.880332, 0.796895},
            {0.962866, 0.528193, 0.561334}, {0.962866, 0.528193, 0.561334},
            {0.742035, 0.885688, 0.458892}, {0.742035, 0.885688, 0.458892},
            {0.383097, 0.761093, 0.173810}, {0.284898, 0.359292, 0.669062}};
    mesh_gt.triangles_ = {
            {1, 13, 0},   {0, 14, 2},   {13, 14, 0},  {1, 15, 13},
            {15, 16, 13}, {1, 3, 15},   {14, 13, 16}, {16, 17, 14},
            {2, 18, 4},   {14, 18, 2},  {4, 18, 5},   {18, 14, 17},
            {17, 19, 18}, {18, 19, 6},  {6, 5, 18},   {7, 16, 15},
            {7, 8, 16},   {8, 20, 16},  {21, 16, 20}, {17, 16, 21},
            {9, 20, 8},   {10, 20, 9},  {21, 20, 10}, {22, 17, 21},
            {19, 17, 22}, {19, 22, 11}, {11, 6, 19},  {22, 21, 10},
            {10, 12, 22}, {11, 22, 12}, {24, 0, 23},  {1, 0, 24},
            {0, 2, 25},   {25, 23, 0},  {3, 1, 24},   {24, 26, 3},
            {2, 4, 27},   {27, 25, 2},  {27, 5, 28},  {4, 5, 27},
            {28, 6, 29},  {5, 6, 28},   {8, 7, 30},   {30, 31, 8},
            {32, 8, 31},  {9, 8, 32},   {33, 9, 32},  {10, 9, 33},
            {6, 11, 34},  {34, 29, 6},  {12, 10, 33}, {33, 35, 12},
            {34, 12, 35}, {11, 12, 34}, {24, 23, 48}, {36, 23, 25},
            {36, 37, 23}, {37, 48, 23}, {38, 24, 48}, {38, 39, 24},
            {39, 26, 24}, {39, 49, 26}, {38, 48, 37}, {25, 27, 40},
            {40, 36, 25}, {27, 28, 41}, {41, 40, 27}, {28, 29, 42},
            {42, 41, 28}, {39, 30, 49}, {39, 31, 30}, {39, 50, 31},
            {39, 43, 50}, {44, 50, 43}, {32, 31, 50}, {44, 32, 50},
            {44, 45, 32}, {45, 33, 32}, {42, 34, 46}, {29, 34, 42},
            {47, 33, 45}, {35, 33, 47}, {34, 35, 47}, {47, 46, 34},
            {37, 36, 51}, {39, 38, 52}, {53, 52, 51}, {52, 37, 51},
            {52, 38, 37}, {36, 40, 54}, {54, 51, 36}, {54, 40, 41},
            {51, 54, 55}, {55, 53, 51}, {55, 41, 42}, {54, 41, 55},
            {43, 39, 52}, {56, 52, 53}, {56, 44, 52}, {44, 43, 52},
            {45, 44, 56}, {56, 55, 57}, {53, 55, 56}, {55, 42, 46},
            {46, 57, 55}, {45, 56, 57}, {57, 47, 45}, {57, 46, 47}};
    std::vector<double> densities_gt = {
            0.39865168929100037, 0.32580316066741943, 0.39778709411621094,
            0.2200755625963211,  0.428702175617218,   0.4288075268268585,
            0.44350555539131165, 0.24208465218544006, 0.3794262409210205,
            0.407772421836853,   0.41914406418800354, 0.4330996870994568,
            0.4330378770828247,  0.3689049780368805,  0.4012255072593689,
            0.07905912399291992, 0.30074167251586914, 0.3844510018825531,
            0.4286557137966156,  0.43353089690208435, 0.3970388174057007,
            0.41389259696006775, 0.4331146478652954,  0.39110293984413147,
            0.3399316370487213,  0.3989846110343933,  0.294955313205719,
            0.44293972849845886, 0.4377133548259735,  0.36585745215415955,
            0.31271663308143616, 0.3885934352874756,  0.4121553897857666,
            0.3849177360534668,  0.3899257779121399,  0.3932742774486542,
            0.42859214544296265, 0.44281646609306335, 0.4422561824321747,
            0.4249078631401062,  0.4210644066333771,  0.41715115308761597,
            0.38326939940452576, 0.43757864832878113, 0.4378965198993683,
            0.42082998156547546, 0.41933944821357727, 0.42220038175582886,
            0.439664751291275,   0.3250156342983246,  0.437633752822876,
            0.4227336645126343,  0.42734524607658386, 0.35958021879196167,
            0.41651853919029236, 0.3831002116203308,  0.41637951135635376,
            0.42157289385795593};

    std::shared_ptr<geometry::TriangleMesh> mesh_es;
    std::vector<double> densities_es;
#if __APPLE__
    // TODO: To be investigated.
    //
    // macOS could sometimes be stuck on this test. Examples:
    // - https://github.com/intel-isl/Open3D/runs/844549493#step:6:3150
    // - https://github.com/intel-isl/Open3D/runs/741891346#step:5:3146
    // - https://github.com/intel-isl/Open3D/runs/734021844#step:5:3169
    //
    // We suspect that this is related to threading. Here we set n_threads=1,
    // and if the macOS CI still stuck on this test occasionally, we might need
    // to look somewhere else.
    std::tie(mesh_es, densities_es) =
            geometry::TriangleMesh::CreateFromPointCloudPoisson(
                    pcd, 2, 0, 1.1f, false, /*n_threads=*/1);
#else
    std::tie(mesh_es, densities_es) =
            geometry::TriangleMesh::CreateFromPointCloudPoisson(pcd, 2);
#endif

    ExpectMeshEQ(*mesh_es, mesh_gt, 1e-4);
    ExpectEQ(densities_es, densities_gt, 1e-4);
}

TEST(TriangleMesh, CreateFromPointCloudAlphaShape) {
    geometry::PointCloud pcd;
    pcd.points_ = {
            {0.765822, 1.000000, 0.486627}, {0.034963, 1.000000, 0.632086},
            {0.000000, 0.093962, 0.028012}, {0.000000, 0.910057, 0.049732},
            {0.017178, 0.000000, 0.946382}, {0.972485, 0.000000, 0.431460},
            {0.794109, 0.033417, 1.000000}, {0.700868, 0.648112, 1.000000},
            {0.164379, 0.516339, 1.000000}, {0.521248, 0.377170, 0.000000}};
    geometry::TriangleMesh mesh_gt;
    mesh_gt.vertices_ = {
            {0.521248, 0.377170, 0.000000}, {0.017178, 0.000000, 0.946382},
            {0.972485, 0.000000, 0.431460}, {0.000000, 0.093962, 0.028012},
            {0.164379, 0.516339, 1.000000}, {0.700868, 0.648112, 1.000000},
            {0.765822, 1.000000, 0.486627}, {0.794109, 0.033417, 1.000000},
            {0.034963, 1.000000, 0.632086}, {0.000000, 0.910057, 0.049732}};
    mesh_gt.triangles_ = {{0, 2, 3}, {1, 2, 3}, {2, 5, 6}, {0, 2, 6},
                          {4, 5, 7}, {2, 5, 7}, {1, 2, 7}, {1, 4, 7},
                          {1, 4, 8}, {1, 3, 8}, {4, 5, 8}, {5, 6, 8},
                          {3, 8, 9}, {0, 3, 9}, {6, 8, 9}, {0, 6, 9}};

    auto mesh_es =
            geometry::TriangleMesh::CreateFromPointCloudAlphaShape(pcd, 1);
    ExpectMeshEQ(*mesh_es, mesh_gt);
}

TEST(TriangleMesh, CreateMeshSphere) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, -1.000000},
            {0.587785, 0.000000, 0.809017},
            {0.475528, 0.345492, 0.809017},
            {0.181636, 0.559017, 0.809017},
            {-0.181636, 0.559017, 0.809017},
            {-0.475528, 0.345492, 0.809017},
            {-0.587785, 0.000000, 0.809017},
            {-0.475528, -0.345492, 0.809017},
            {-0.181636, -0.559017, 0.809017},
            {0.181636, -0.559017, 0.809017},
            {0.475528, -0.345492, 0.809017},
            {0.951057, 0.000000, 0.309017},
            {0.769421, 0.559017, 0.309017},
            {0.293893, 0.904508, 0.309017},
            {-0.293893, 0.904508, 0.309017},
            {-0.769421, 0.559017, 0.309017},
            {-0.951057, 0.000000, 0.309017},
            {-0.769421, -0.559017, 0.309017},
            {-0.293893, -0.904508, 0.309017},
            {0.293893, -0.904508, 0.309017},
            {0.769421, -0.559017, 0.309017},
            {0.951057, 0.000000, -0.309017},
            {0.769421, 0.559017, -0.309017},
            {0.293893, 0.904508, -0.309017},
            {-0.293893, 0.904508, -0.309017},
            {-0.769421, 0.559017, -0.309017},
            {-0.951057, 0.000000, -0.309017},
            {-0.769421, -0.559017, -0.309017},
            {-0.293893, -0.904508, -0.309017},
            {0.293893, -0.904508, -0.309017},
            {0.769421, -0.559017, -0.309017},
            {0.587785, 0.000000, -0.809017},
            {0.475528, 0.345492, -0.809017},
            {0.181636, 0.559017, -0.809017},
            {-0.181636, 0.559017, -0.809017},
            {-0.475528, 0.345492, -0.809017},
            {-0.587785, 0.000000, -0.809017},
            {-0.475528, -0.345492, -0.809017},
            {-0.181636, -0.559017, -0.809017},
            {0.181636, -0.559017, -0.809017},
            {0.475528, -0.345492, -0.809017}};

    std::vector<Eigen::Vector3i> ref_triangles = {
            {0, 2, 3},    {1, 33, 32},  {0, 3, 4},    {1, 34, 33},
            {0, 4, 5},    {1, 35, 34},  {0, 5, 6},    {1, 36, 35},
            {0, 6, 7},    {1, 37, 36},  {0, 7, 8},    {1, 38, 37},
            {0, 8, 9},    {1, 39, 38},  {0, 9, 10},   {1, 40, 39},
            {0, 10, 11},  {1, 41, 40},  {0, 11, 2},   {1, 32, 41},
            {12, 3, 2},   {12, 13, 3},  {13, 4, 3},   {13, 14, 4},
            {14, 5, 4},   {14, 15, 5},  {15, 6, 5},   {15, 16, 6},
            {16, 7, 6},   {16, 17, 7},  {17, 8, 7},   {17, 18, 8},
            {18, 9, 8},   {18, 19, 9},  {19, 10, 9},  {19, 20, 10},
            {20, 11, 10}, {20, 21, 11}, {21, 2, 11},  {21, 12, 2},
            {22, 13, 12}, {22, 23, 13}, {23, 14, 13}, {23, 24, 14},
            {24, 15, 14}, {24, 25, 15}, {25, 16, 15}, {25, 26, 16},
            {26, 17, 16}, {26, 27, 17}, {27, 18, 17}, {27, 28, 18},
            {28, 19, 18}, {28, 29, 19}, {29, 20, 19}, {29, 30, 20},
            {30, 21, 20}, {30, 31, 21}, {31, 12, 21}, {31, 22, 12},
            {32, 23, 22}, {32, 33, 23}, {33, 24, 23}, {33, 34, 24},
            {34, 25, 24}, {34, 35, 25}, {35, 26, 25}, {35, 36, 26},
            {36, 27, 26}, {36, 37, 27}, {37, 28, 27}, {37, 38, 28},
            {38, 29, 28}, {38, 39, 29}, {39, 30, 29}, {39, 40, 30},
            {40, 31, 30}, {40, 41, 31}, {41, 22, 31}, {41, 32, 22}};

    auto output_tm = geometry::TriangleMesh::CreateSphere(1.0, 5);

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
}

TEST(TriangleMesh, CreateMeshCylinder) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, -1.000000},
            {1.000000, 0.000000, 1.000000},
            {0.309017, 0.951057, 1.000000},
            {-0.809017, 0.587785, 1.000000},
            {-0.809017, -0.587785, 1.000000},
            {0.309017, -0.951057, 1.000000},
            {1.000000, 0.000000, 0.500000},
            {0.309017, 0.951057, 0.500000},
            {-0.809017, 0.587785, 0.500000},
            {-0.809017, -0.587785, 0.500000},
            {0.309017, -0.951057, 0.500000},
            {1.000000, 0.000000, 0.000000},
            {0.309017, 0.951057, 0.000000},
            {-0.809017, 0.587785, 0.000000},
            {-0.809017, -0.587785, 0.000000},
            {0.309017, -0.951057, 0.000000},
            {1.000000, 0.000000, -0.500000},
            {0.309017, 0.951057, -0.500000},
            {-0.809017, 0.587785, -0.500000},
            {-0.809017, -0.587785, -0.500000},
            {0.309017, -0.951057, -0.500000},
            {1.000000, 0.000000, -1.000000},
            {0.309017, 0.951057, -1.000000},
            {-0.809017, 0.587785, -1.000000},
            {-0.809017, -0.587785, -1.000000},
            {0.309017, -0.951057, -1.000000}};

    std::vector<Eigen::Vector3i> ref_triangles = {
            {0, 2, 3},    {1, 23, 22},  {0, 3, 4},    {1, 24, 23},
            {0, 4, 5},    {1, 25, 24},  {0, 5, 6},    {1, 26, 25},
            {0, 6, 2},    {1, 22, 26},  {7, 3, 2},    {7, 8, 3},
            {8, 4, 3},    {8, 9, 4},    {9, 5, 4},    {9, 10, 5},
            {10, 6, 5},   {10, 11, 6},  {11, 2, 6},   {11, 7, 2},
            {12, 8, 7},   {12, 13, 8},  {13, 9, 8},   {13, 14, 9},
            {14, 10, 9},  {14, 15, 10}, {15, 11, 10}, {15, 16, 11},
            {16, 7, 11},  {16, 12, 7},  {17, 13, 12}, {17, 18, 13},
            {18, 14, 13}, {18, 19, 14}, {19, 15, 14}, {19, 20, 15},
            {20, 16, 15}, {20, 21, 16}, {21, 12, 16}, {21, 17, 12},
            {22, 18, 17}, {22, 23, 18}, {23, 19, 18}, {23, 24, 19},
            {24, 20, 19}, {24, 25, 20}, {25, 21, 20}, {25, 26, 21},
            {26, 17, 21}, {26, 22, 17}};

    auto output_tm = geometry::TriangleMesh::CreateCylinder(1.0, 2.0, 5);

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
}

TEST(TriangleMesh, CreateMeshCone) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {0.000000, 0.000000, 0.000000},  {0.000000, 0.000000, 2.000000},
            {1.000000, 0.000000, 0.000000},  {0.309017, 0.951057, 0.000000},
            {-0.809017, 0.587785, 0.000000}, {-0.809017, -0.587785, 0.000000},
            {0.309017, -0.951057, 0.000000}};

    std::vector<Eigen::Vector3i> ref_triangles = {
            {0, 3, 2}, {1, 2, 3}, {0, 4, 3}, {1, 3, 4}, {0, 5, 4},
            {1, 4, 5}, {0, 6, 5}, {1, 5, 6}, {0, 2, 6}, {1, 6, 2}};

    auto output_tm = geometry::TriangleMesh::CreateCone(1.0, 2.0, 5);

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
}

TEST(TriangleMesh, CreateMeshArrow) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {0.000000, 0.000000, 2.000000},   {0.000000, 0.000000, 0.000000},
            {1.000000, 0.000000, 2.000000},   {0.309017, 0.951057, 2.000000},
            {-0.809017, 0.587785, 2.000000},  {-0.809017, -0.587785, 2.000000},
            {0.309017, -0.951057, 2.000000},  {1.000000, 0.000000, 1.500000},
            {0.309017, 0.951057, 1.500000},   {-0.809017, 0.587785, 1.500000},
            {-0.809017, -0.587785, 1.500000}, {0.309017, -0.951057, 1.500000},
            {1.000000, 0.000000, 1.000000},   {0.309017, 0.951057, 1.000000},
            {-0.809017, 0.587785, 1.000000},  {-0.809017, -0.587785, 1.000000},
            {0.309017, -0.951057, 1.000000},  {1.000000, 0.000000, 0.500000},
            {0.309017, 0.951057, 0.500000},   {-0.809017, 0.587785, 0.500000},
            {-0.809017, -0.587785, 0.500000}, {0.309017, -0.951057, 0.500000},
            {1.000000, 0.000000, 0.000000},   {0.309017, 0.951057, 0.000000},
            {-0.809017, 0.587785, 0.000000},  {-0.809017, -0.587785, 0.000000},
            {0.309017, -0.951057, 0.000000},  {0.000000, 0.000000, 2.000000},
            {0.000000, 0.000000, 3.000000},   {1.500000, 0.000000, 2.000000},
            {0.463525, 1.426585, 2.000000},   {-1.213525, 0.881678, 2.000000},
            {-1.213525, -0.881678, 2.000000}, {0.463525, -1.426585, 2.000000}};

    std::vector<Eigen::Vector3i> ref_triangles = {
            {0, 2, 3},    {1, 23, 22},  {0, 3, 4},    {1, 24, 23},
            {0, 4, 5},    {1, 25, 24},  {0, 5, 6},    {1, 26, 25},
            {0, 6, 2},    {1, 22, 26},  {7, 3, 2},    {7, 8, 3},
            {8, 4, 3},    {8, 9, 4},    {9, 5, 4},    {9, 10, 5},
            {10, 6, 5},   {10, 11, 6},  {11, 2, 6},   {11, 7, 2},
            {12, 8, 7},   {12, 13, 8},  {13, 9, 8},   {13, 14, 9},
            {14, 10, 9},  {14, 15, 10}, {15, 11, 10}, {15, 16, 11},
            {16, 7, 11},  {16, 12, 7},  {17, 13, 12}, {17, 18, 13},
            {18, 14, 13}, {18, 19, 14}, {19, 15, 14}, {19, 20, 15},
            {20, 16, 15}, {20, 21, 16}, {21, 12, 16}, {21, 17, 12},
            {22, 18, 17}, {22, 23, 18}, {23, 19, 18}, {23, 24, 19},
            {24, 20, 19}, {24, 25, 20}, {25, 21, 20}, {25, 26, 21},
            {26, 17, 21}, {26, 22, 17}, {27, 30, 29}, {28, 29, 30},
            {27, 31, 30}, {28, 30, 31}, {27, 32, 31}, {28, 31, 32},
            {27, 33, 32}, {28, 32, 33}, {27, 29, 33}, {28, 33, 29}};

    auto output_tm = geometry::TriangleMesh::CreateArrow(1.0, 1.5, 2.0, 1.0, 5);

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
}

TEST(TriangleMesh, CreateMeshCoordinateFrame) {
    std::vector<Eigen::Vector3d> ref_vertices = {
            {0, 0, 0.006},
            {0.000938607, 0, 0.00592613},
            {0.000927051, 0.00014683, 0.00592613},
            {-5.0038e-19, -0.00272394, 0.00534604},
            {0.000426119, -0.00269041, 0.00534604},
            {-6.47846e-19, -0.00352671, 0.0048541},
            {-0.00534604, 6.54701e-19, 0.00272394},
            {-0.00528022, -0.000836305, 0.00272394},
            {-0.00570634, 6.98825e-19, 0.0018541},
            {3.62871e-19, 0.00592613, -0.000938607},
            {-0.000927051, 0.00585317, -0.000938607},
            {3.49412e-19, 0.00570634, -0.0018541},
            {0.00424264, 0, -0.00424264},
            {0.00419041, 0.000663695, -0.00424264},
            {0.00352671, 0, -0.0048541},
            {-3.40593e-19, -0.0018541, -0.00570634},
            {0.000290045, -0.00183127, -0.00570634},
            {-1.72419e-19, -0.000938607, -0.00592613},
            {0, 0.08, 0.0035},
            {0.00108156, 0.08, 0.0033287},
            {0, 0.06, 0.0035},
            {-0.0035, 4.28626e-19, 0.06},
            {-0.0033287, -0.00108156, 0.06},
            {-0.0035, 4.28626e-19, 0.04}};

    std::vector<Eigen::Vector3d> ref_vertex_normals = {
            {0.000000, 0.000000, 1.000000},   {0.182067, -0.006090, 0.983267},
            {0.180778, 0.022466, 0.983267},   {-0.005496, -0.461174, 0.887292},
            {0.066715, -0.456356, 0.887292},  {-0.004990, -0.592357, 0.805660},
            {-0.891954, 0.002800, 0.452117},  {-0.881411, -0.136766, 0.452117},
            {-0.951468, 0.001906, 0.307743},  {-0.000965, 0.987790, -0.155790},
            {-0.155477, 0.975477, -0.155790}, {-0.001906, 0.951468, -0.307743},
            {0.710008, 0.004362, -0.704180},  {0.700584, 0.115378, -0.704180},
            {0.592357, 0.004990, -0.805660},  {0.005866, -0.321047, -0.947045},
            {0.056017, -0.316177, -0.947045}, {0.006090, -0.182067, -0.983267},
            {-0.052367, 0.115722, 0.991900},  {0.256710, 0.115722, 0.959536},
            {0.000000, 0.000000, 1.000000},   {-1.000000, 0.000000, 0.000000},
            {-0.951057, -0.309017, 0.000000}, {-1.000000, 0.000000, 0.000000}};

    std::vector<Eigen::Vector3d> ref_vertex_colors = {
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.500000, 0.500000, 0.500000}, {0.500000, 0.500000, 0.500000},
            {0.000000, 1.000000, 0.000000}, {0.000000, 1.000000, 0.000000},
            {0.000000, 1.000000, 0.000000}, {0.000000, 0.000000, 1.000000},
            {0.000000, 0.000000, 1.000000}, {0.000000, 0.000000, 1.000000}};

    std::vector<Eigen::Vector3i> ref_triangles = {
            {0, 1, 2},    {5, 4, 3},    {8, 7, 6},    {11, 10, 9},
            {14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}};

    std::vector<Eigen::Vector3d> ref_triangle_normals = {
            {0.078458, 0.006175, 0.996898},   {0.041087, -0.522057, 0.851920},
            {-0.921447, -0.072519, 0.381676}, {-0.076304, 0.969535, -0.232765},
            {0.648601, 0.051046, -0.759415},  {0.018369, -0.233406, -0.972206},
            {0.156434, 0.000000, 0.987688},   {-0.987688, -0.156434, 0.000000}};

    auto output_tm = geometry::TriangleMesh::CreateCoordinateFrame(0.1);

    EXPECT_EQ(1134u, output_tm->vertices_.size());
    EXPECT_EQ(1134u, output_tm->vertex_normals_.size());
    EXPECT_EQ(1134u, output_tm->vertex_colors_.size());
    EXPECT_EQ(2240u, output_tm->triangles_.size());
    EXPECT_EQ(2240u, output_tm->triangle_normals_.size());

    // CreateMeshCoordinateFrame generates too many values
    // down sample to a more manageable size before comparing
    int stride = 300;
    std::vector<size_t> indices;
    for (size_t i = 0; i < output_tm->triangles_.size(); i += stride) {
        indices.push_back(output_tm->triangles_[i](0, 0));
        indices.push_back(output_tm->triangles_[i](1, 0));
        indices.push_back(output_tm->triangles_[i](2, 0));
    }
    unique(indices.begin(), indices.end());
    sort(indices.begin(), indices.end());
    auto output = output_tm->SelectByIndex(indices);

    ExpectEQ(ref_vertices, output->vertices_);
    ExpectEQ(ref_vertex_normals, output->vertex_normals_);
    ExpectEQ(ref_vertex_colors, output->vertex_colors_);
    ExpectEQ(ref_triangles, output->triangles_);
    ExpectEQ(ref_triangle_normals, output->triangle_normals_);

    // Test with different origin
    double size = 1;
    auto center_frame = geometry::TriangleMesh::CreateCoordinateFrame(size);

    Eigen::Vector3d x(1, 0, 0);
    auto x_frame = geometry::TriangleMesh::CreateCoordinateFrame(size, x);
    Eigen::Vector3d x_center = x_frame->GetCenter() - x;
    ExpectEQ(center_frame->GetCenter(), x_center);

    Eigen::Vector3d y(0, 1, 0);
    auto y_frame = geometry::TriangleMesh::CreateCoordinateFrame(size, y);
    Eigen::Vector3d y_center = y_frame->GetCenter() - y;
    ExpectEQ(center_frame->GetCenter(), y_center);

    Eigen::Vector3d z(0, 0, 1);
    auto z_frame = geometry::TriangleMesh::CreateCoordinateFrame(size, z);
    Eigen::Vector3d z_center = z_frame->GetCenter() - z;
    ExpectEQ(center_frame->GetCenter(), z_center);
}

}  // namespace tests
}  // namespace open3d
