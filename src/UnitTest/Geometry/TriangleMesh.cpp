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

#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Geometry/PointCloud.h"
#include "TestUtility/UnitTest.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, DISABLED_MemberData) { unit_test::NotImplemented(); }

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, Clear) {
    int size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

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

    ExpectEQ(Vector3d(19.607843, 0.0, 0.0), tm.GetMinBound());
    ExpectEQ(Vector3d(996.078431, 996.078431, 996.078431), tm.GetMaxBound());

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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, IsEmpty) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_TRUE(tm.IsEmpty());

    tm.vertices_.resize(size);

    EXPECT_FALSE(tm.IsEmpty());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, GetMinBound) {
    int size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    ExpectEQ(Vector3d(19.607843, 0.0, 0.0), tm.GetMinBound());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, GetMaxBound) {
    int size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    ExpectEQ(Vector3d(996.078431, 996.078431, 996.078431), tm.GetMaxBound());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, Transform) {
    vector<Vector3d> ref_vertices = {
            {1.411252, 4.274168, 3.130918}, {1.231757, 4.154505, 3.183678},
            {1.403168, 4.268779, 2.121679}, {1.456767, 4.304511, 2.640845},
            {1.620902, 4.413935, 1.851255}, {1.374684, 4.249790, 3.062485},
            {1.328160, 4.218773, 1.795728}, {1.713446, 4.475631, 1.860145},
            {1.409239, 4.272826, 2.011462}, {1.480169, 4.320113, 1.177780}};

    vector<Vector3d> ref_vertex_normals = {
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

    vector<Vector3d> ref_triangle_normals = {
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

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.triangle_normals_.resize(size);

    Rand(tm.vertices_, dmin, dmax, 0);
    Rand(tm.vertex_normals_, dmin, dmax, 0);
    Rand(tm.triangle_normals_, dmin, dmax, 0);

    Matrix4d transformation;
    transformation << 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16;

    tm.Transform(transformation);

    ExpectEQ(ref_vertices, tm.vertices_);
    ExpectEQ(ref_vertex_normals, tm.vertex_normals_);
    ExpectEQ(ref_triangle_normals, tm.triangle_normals_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, OperatorAppend) {
    size_t size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

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
        ExpectEQ(Vector3i(tm1.triangles_[i](0, 0) + size,
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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, OperatorADD) {
    size_t size = 100;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

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
        ExpectEQ(Vector3i(tm1.triangles_[i](0, 0) + size,
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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, ComputeTriangleNormals) {
    vector<Vector3d> ref = {{-0.119231, 0.738792, 0.663303},
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

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(10.0, 10.0, 10.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    for (size_t i = 0; i < size; i++)
        tm.triangles_.push_back(Vector3i(i, (i + 1) % size, (i + 2) % size));

    tm.ComputeTriangleNormals();

    ExpectEQ(ref, tm.triangle_normals_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, ComputeVertexNormals) {
    vector<Vector3d> ref = {
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

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(10.0, 10.0, 10.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    Rand(tm.vertices_, dmin, dmax, 0);

    for (size_t i = 0; i < size; i++)
        tm.triangles_.push_back(Vector3i(i, (i + 1) % size, (i + 2) % size));

    tm.ComputeVertexNormals();

    ExpectEQ(ref, tm.vertex_normals_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, Purge) {
    vector<Vector3d> ref_vertices = {{839.215686, 392.156863, 780.392157},
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

    vector<Vector3d> ref_vertex_normals = {{839.215686, 392.156863, 780.392157},
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

    vector<Vector3d> ref_vertex_colors = {{839.215686, 392.156863, 780.392157},
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

    vector<Vector3i> ref_triangles = {
            {20, 9, 18},  {19, 21, 4}, {8, 18, 6}, {13, 11, 15}, {8, 12, 22},
            {21, 15, 17}, {3, 14, 0},  {5, 3, 19}, {2, 23, 5},   {12, 20, 14},
            {7, 15, 12},  {11, 23, 6}, {9, 21, 6}, {8, 19, 22},  {1, 22, 12},
            {1, 4, 15},   {21, 8, 1},  {0, 10, 1}, {5, 23, 21},  {20, 6, 12},
            {8, 18, 12},  {16, 12, 0}};

    vector<Vector3d> ref_triangle_normals = {
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

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, SamplePointsUniformly) {
    auto mesh_empty = geometry::TriangleMesh();
    auto pcd_empty = mesh_empty.SamplePointsUniformly(100);
    EXPECT_TRUE(pcd_empty->points_.size() == 0);
    EXPECT_TRUE(pcd_empty->colors_.size() == 0);
    EXPECT_TRUE(pcd_empty->normals_.size() == 0);

    vector<Vector3d> vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    vector<Vector3i> triangles = {{0, 1, 2}};

    auto mesh_simple = geometry::TriangleMesh();
    mesh_simple.vertices_ = vertices;
    mesh_simple.triangles_ = triangles;

    size_t n_points = 100;
    auto pcd_simple = mesh_simple.SamplePointsUniformly(n_points);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == 0);
    EXPECT_TRUE(pcd_simple->normals_.size() == 0);

    vector<Vector3d> colors = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
    vector<Vector3d> normals = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
    mesh_simple.vertex_colors_ = colors;
    mesh_simple.vertex_normals_ = normals;
    pcd_simple = mesh_simple.SamplePointsUniformly(n_points);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(pcd_simple->colors_[pidx], Vector3d(1, 0, 0));
        ExpectEQ(pcd_simple->normals_[pidx], Vector3d(0, 1, 0));
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
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
    ExpectEQ(mesh->vertices_, ref1);

    mesh = mesh->FilterSmoothSimple(3);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.003906, 0, 0},
                                         {0, 0.003906, 0},
                                         {-0.003906, 0, 0},
                                         {0, -0.003906, 0}};
    ExpectEQ(mesh->vertices_, ref2);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, FilterSmoothLaplacian) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = mesh->FilterSmoothLaplacian(1, 0.5);
    std::vector<Eigen::Vector3d> ref1 = {
            {0, 0, 0}, {0.5, 0, 0}, {0, 0.5, 0}, {-0.5, 0, 0}, {0, -0.5, 0}};
    ExpectEQ(mesh->vertices_, ref1);

    mesh = mesh->FilterSmoothLaplacian(10, 0.5);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.000488, 0, 0},
                                         {0, 0.000488, 0},
                                         {-0.000488, 0, 0},
                                         {0, -0.000488, 0}};
    ExpectEQ(mesh->vertices_, ref2);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
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
    ExpectEQ(mesh->vertices_, ref1);

    mesh = mesh->FilterSmoothTaubin(10, 0.5, -0.53);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.052514, 0, 0},
                                         {0, 0.052514, 0},
                                         {-0.052514, 0, 0},
                                         {0, -0.052514, 0}};
    ExpectEQ(mesh->vertices_, ref2);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasVertices) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertices());

    tm.vertices_.resize(size);

    EXPECT_TRUE(tm.HasVertices());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasTriangles) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangles());

    tm.vertices_.resize(size);
    tm.triangles_.resize(size);

    EXPECT_TRUE(tm.HasTriangles());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasVertexNormals) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexNormals());

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);

    EXPECT_TRUE(tm.HasVertexNormals());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasVertexColors) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexColors());

    tm.vertices_.resize(size);
    tm.vertex_colors_.resize(size);

    EXPECT_TRUE(tm.HasVertexColors());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasTriangleNormals) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangleNormals());

    tm.vertices_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    EXPECT_TRUE(tm.HasTriangleNormals());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, NormalizeNormals) {
    vector<Vector3d> ref_vertex_normals = {
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

    vector<Vector3d> ref_triangle_normals = {
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

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(10.0, 10.0, 10.0);

    geometry::TriangleMesh tm;

    tm.vertex_normals_.resize(size);
    tm.triangle_normals_.resize(size);
    Rand(tm.vertex_normals_, dmin, dmax, 0);
    Rand(tm.triangle_normals_, dmin, dmax, 1);

    tm.NormalizeNormals();

    ExpectEQ(ref_vertex_normals, tm.vertex_normals_);
    ExpectEQ(ref_triangle_normals, tm.triangle_normals_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, PaintUniformColor) {
    int size = 25;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(10.0, 10.0, 10.0);

    geometry::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_colors_.resize(size);

    Vector3d color(233. / 255., 171. / 255., 53.0 / 255.);
    tm.PaintUniformColor(color);

    for (size_t i = 0; i < tm.vertex_colors_.size(); i++)
        ExpectEQ(color, tm.vertex_colors_[i]);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, EulerPoincareCharacteristic) {
    EXPECT_EQ(geometry::TriangleMesh::CreateBox()
                              ->EulerPoincareCharacteristic() == 2,
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateSphere()
                              ->EulerPoincareCharacteristic() == 2,
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateCylinder()
                              ->EulerPoincareCharacteristic() == 2,
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateCone()
                              ->EulerPoincareCharacteristic() == 2,
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateTorus()
                              ->EulerPoincareCharacteristic() == 0,
              true);

    geometry::TriangleMesh mesh0;
    mesh0.vertices_ = {{0, 0, 0},  {1, 0, 0},  {1, 1, 0}, {1, 1, 1},
                       {-1, 0, 0}, {-1, 1, 0}, {-1, 1, 1}};
    mesh0.triangles_ = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
                        {0, 4, 5}, {0, 4, 6}, {0, 5, 6}, {4, 5, 6}};
    EXPECT_EQ(mesh0.EulerPoincareCharacteristic() == 3, true);

    geometry::TriangleMesh mesh1;
    mesh1.vertices_ = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 0, 2}, {1, 0.5, 1}};
    mesh1.triangles_ = {{0, 1, 2}, {1, 2, 3}, {1, 2, 4}};
    EXPECT_EQ(mesh1.EulerPoincareCharacteristic() == 1, true);
}

TEST(TriangleMesh, IsEdgeManifold) {
    EXPECT_EQ(geometry::TriangleMesh::CreateBox()->IsEdgeManifold(true), true);
    EXPECT_EQ(geometry::TriangleMesh::CreateSphere()->IsEdgeManifold(true),
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateCylinder()->IsEdgeManifold(true),
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateCone()->IsEdgeManifold(true), true);
    EXPECT_EQ(geometry::TriangleMesh::CreateTorus()->IsEdgeManifold(true),
              true);

    EXPECT_EQ(geometry::TriangleMesh::CreateBox()->IsEdgeManifold(false), true);
    EXPECT_EQ(geometry::TriangleMesh::CreateSphere()->IsEdgeManifold(false),
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateCylinder()->IsEdgeManifold(false),
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateCone()->IsEdgeManifold(false),
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateTorus()->IsEdgeManifold(false),
              true);

    geometry::TriangleMesh mesh0;
    mesh0.vertices_ = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 0, 2}, {1, 0.5, 1}};
    mesh0.triangles_ = {{0, 1, 2}, {1, 2, 3}, {1, 2, 4}};
    EXPECT_EQ(mesh0.IsEdgeManifold(true), false);
    EXPECT_EQ(mesh0.IsEdgeManifold(false), false);

    geometry::TriangleMesh mesh1;
    mesh1.vertices_ = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 0, 2}};
    mesh1.triangles_ = {{0, 1, 2}, {1, 2, 3}};
    EXPECT_EQ(mesh1.IsEdgeManifold(true), true);
    EXPECT_EQ(mesh1.IsEdgeManifold(false), false);
}

TEST(TriangleMesh, IsVertexManifold) {
    EXPECT_EQ(geometry::TriangleMesh::CreateBox()->IsVertexManifold(), true);
    EXPECT_EQ(geometry::TriangleMesh::CreateSphere()->IsVertexManifold(), true);
    EXPECT_EQ(geometry::TriangleMesh::CreateCylinder()->IsVertexManifold(),
              true);
    EXPECT_EQ(geometry::TriangleMesh::CreateCone()->IsVertexManifold(), true);
    EXPECT_EQ(geometry::TriangleMesh::CreateTorus()->IsVertexManifold(), true);

    geometry::TriangleMesh mesh0;
    mesh0.vertices_ = {{0, 0, 0}, {1, 1, 1},  {1, 0, 1},
                       {0, 1, 1}, {1, 1, -1}, {1, 0, -1}};
    mesh0.triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 4, 5}};
    EXPECT_EQ(mesh0.IsVertexManifold(), false);

    geometry::TriangleMesh mesh1;
    mesh1.vertices_ = {{0, 0, 0},  {1, 1, 1},  {1, 0, 1}, {0, 1, 1},
                       {1, 1, -1}, {1, 0, -1}, {0, 1, -1}};
    mesh1.triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 4, 5}, {0, 5, 6}};
    EXPECT_EQ(mesh1.IsVertexManifold(), false);
}

TEST(TriangleMesh, IsSelfIntersecting) {
    EXPECT_EQ(geometry::TriangleMesh::CreateBox()->IsSelfIntersecting(), false);
    EXPECT_EQ(geometry::TriangleMesh::CreateSphere()->IsSelfIntersecting(),
              false);
    EXPECT_EQ(geometry::TriangleMesh::CreateCylinder()->IsSelfIntersecting(),
              false);
    EXPECT_EQ(geometry::TriangleMesh::CreateCone()->IsSelfIntersecting(),
              false);
    EXPECT_EQ(geometry::TriangleMesh::CreateTorus()->IsSelfIntersecting(),
              false);

    // simple intersection
    geometry::TriangleMesh mesh0;
    mesh0.vertices_ = {{0, 0, 0},      {0, 1, 0}, {1, 0, 0}, {1, 1, 0},
                       {0.5, 0.5, -1}, {0, 1, 1}, {1, 0, 1}};
    mesh0.triangles_ = {{0, 1, 2}, {1, 2, 3}, {4, 5, 6}};
    EXPECT_EQ(mesh0.IsSelfIntersecting(), true);

    // co-planar intersection
    geometry::TriangleMesh mesh1;
    mesh1.vertices_ = {{0, 0, 0},     {0, 1, 0},     {1, 0, 0},
                       {0.1, 0.1, 0}, {0.1, 1.1, 0}, {1.1, 0.1, 0}};
    mesh1.triangles_ = {{0, 1, 2}, {3, 4, 5}};
    EXPECT_EQ(mesh1.IsSelfIntersecting(), true);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, SelectDownSample) {
    vector<Vector3d> ref_vertices = {{349.019608, 803.921569, 917.647059},
                                     {439.215686, 117.647059, 588.235294},
                                     {290.196078, 356.862745, 874.509804},
                                     {258.823529, 647.058824, 549.019608},
                                     {764.705882, 333.333333, 533.333333},
                                     {886.274510, 945.098039, 54.901961},
                                     {600.000000, 129.411765, 78.431373},
                                     {529.411765, 454.901961, 639.215686},
                                     {39.215686, 435.294118, 929.411765},
                                     {90.196078, 133.333333, 517.647059},
                                     {588.235294, 576.470588, 529.411765},
                                     {866.666667, 545.098039, 737.254902},
                                     {447.058824, 486.274510, 792.156863},
                                     {90.196078, 960.784314, 866.666667},
                                     {47.058824, 525.490196, 176.470588},
                                     {831.372549, 713.725490, 631.372549},
                                     {811.764706, 682.352941, 909.803922},
                                     {39.215686, 411.764706, 694.117647},
                                     {913.725490, 858.823529, 200.000000},
                                     {647.058824, 360.784314, 286.274510},
                                     {635.294118, 984.313725, 337.254902},
                                     {360.784314, 717.647059, 800.000000},
                                     {811.764706, 752.941176, 352.941176},
                                     {341.176471, 47.058824, 764.705882},
                                     {945.098039, 98.039216, 274.509804}};

    vector<Vector3d> ref_vertex_normals = {{803.921569, 917.647059, 66.666667},
                                           {117.647059, 588.235294, 576.470588},
                                           {356.862745, 874.509804, 341.176471},
                                           {647.058824, 549.019608, 917.647059},
                                           {333.333333, 533.333333, 215.686275},
                                           {945.098039, 54.901961, 921.568627},
                                           {129.411765, 78.431373, 376.470588},
                                           {454.901961, 639.215686, 713.725490},
                                           {435.294118, 929.411765, 929.411765},
                                           {133.333333, 517.647059, 74.509804},
                                           {576.470588, 529.411765, 592.156863},
                                           {545.098039, 737.254902, 929.411765},
                                           {486.274510, 792.156863, 635.294118},
                                           {960.784314, 866.666667, 164.705882},
                                           {525.490196, 176.470588, 925.490196},
                                           {713.725490, 631.372549, 513.725490},
                                           {682.352941, 909.803922, 482.352941},
                                           {411.764706, 694.117647, 670.588235},
                                           {858.823529, 200.000000, 792.156863},
                                           {360.784314, 286.274510, 329.411765},
                                           {984.313725, 337.254902, 894.117647},
                                           {717.647059, 800.000000, 674.509804},
                                           {752.941176, 352.941176, 996.078431},
                                           {47.058824, 764.705882, 800.000000},
                                           {98.039216, 274.509804, 239.215686}};

    vector<Vector3d> ref_vertex_colors = {{654.901961, 439.215686, 396.078431},
                                          {94.117647, 945.098039, 274.509804},
                                          {125.490196, 82.352941, 784.313725},
                                          {141.176471, 560.784314, 870.588235},
                                          {756.862745, 133.333333, 74.509804},
                                          {168.627451, 525.490196, 596.078431},
                                          {239.215686, 341.176471, 921.568627},
                                          {43.137255, 772.549020, 858.823529},
                                          {172.549020, 796.078431, 654.901961},
                                          {160.784314, 862.745098, 462.745098},
                                          {274.509804, 807.843137, 745.098039},
                                          {513.725490, 341.176471, 94.117647},
                                          {635.294118, 184.313725, 623.529412},
                                          {450.980392, 411.764706, 403.921569},
                                          {47.058824, 800.000000, 678.431373},
                                          {266.666667, 329.411765, 756.862745},
                                          {831.372549, 870.588235, 976.470588},
                                          {882.352941, 156.862745, 827.450980},
                                          {882.352941, 180.392157, 776.470588},
                                          {200.000000, 698.039216, 886.274510},
                                          {203.921569, 400.000000, 678.431373},
                                          {396.078431, 392.156863, 596.078431},
                                          {286.274510, 90.196078, 933.333333},
                                          {337.254902, 133.333333, 3.921569},
                                          {764.705882, 533.333333, 474.509804}};

    vector<Vector3i> ref_triangles = {
            {19, 7, 12},  {8, 23, 2},  {6, 14, 24},  {16, 0, 20},  {22, 17, 10},
            {11, 15, 13}, {21, 19, 7}, {12, 8, 23},  {2, 5, 3},    {4, 6, 14},
            {24, 16, 0},  {9, 22, 17}, {10, 11, 15}, {3, 8, 10},   {7, 12, 8},
            {23, 2, 5},   {18, 4, 6},  {14, 24, 16}, {12, 10, 23}, {17, 10, 11},
            {15, 13, 1}};

    vector<Vector3d> ref_triangle_normals = {
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

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

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

    vector<size_t> indices(size / 40);
    Rand(indices, 0, size - 1, 0);

    auto output_tm = tm.SelectDownSample(indices);

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_vertex_normals, output_tm->vertex_normals_);
    ExpectEQ(ref_vertex_colors, output_tm->vertex_colors_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
    ExpectEQ(ref_triangle_normals, output_tm->triangle_normals_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CropTriangleMesh) {
    vector<Vector3d> ref_vertices = {{615.686275, 639.215686, 517.647059},
                                     {615.686275, 760.784314, 772.549020},
                                     {678.431373, 643.137255, 443.137255},
                                     {345.098039, 529.411765, 454.901961},
                                     {360.784314, 576.470588, 592.156863},
                                     {317.647059, 666.666667, 525.490196}};

    vector<Vector3d> ref_vertex_normals = {
            {639.215686, 517.647059, 400.000000},
            {760.784314, 772.549020, 282.352941},
            {643.137255, 443.137255, 266.666667},
            {529.411765, 454.901961, 639.215686},
            {576.470588, 592.156863, 662.745098},
            {666.666667, 525.490196, 847.058824}};

    vector<Vector3d> ref_vertex_colors = {{647.058824, 325.490196, 603.921569},
                                          {941.176471, 121.568627, 513.725490},
                                          {807.843137, 309.803922, 3.921569},
                                          {686.274510, 43.137255, 772.549020},
                                          {576.470588, 698.039216, 831.372549},
                                          {854.901961, 341.176471, 878.431373}};

    vector<Vector3i> ref_triangles = {{1, 0, 3}, {5, 2, 4}};

    vector<Vector3d> ref_triangle_normals = {
            {125.490196, 160.784314, 913.725490},
            {764.705882, 901.960784, 807.843137}};

    int size = 1000;

    Vector3d dmin(0.0, 0.0, 0.0);
    Vector3d dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

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

    Vector3d cropBoundMin(300.0, 300.0, 300.0);
    Vector3d cropBoundMax(800.0, 800.0, 800.0);

    auto output_tm = tm.Crop(cropBoundMin, cropBoundMax);

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_vertex_normals, output_tm->vertex_normals_);
    ExpectEQ(ref_vertex_colors, output_tm->vertex_colors_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
    ExpectEQ(ref_triangle_normals, output_tm->triangle_normals_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshSphere) {
    vector<Vector3d> ref_vertices = {{0.000000, 0.000000, 1.000000},
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

    vector<Vector3i> ref_triangles = {
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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshCylinder) {
    vector<Vector3d> ref_vertices = {{0.000000, 0.000000, 1.000000},
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

    vector<Vector3i> ref_triangles = {
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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshCone) {
    vector<Vector3d> ref_vertices = {
            {0.000000, 0.000000, 0.000000},  {0.000000, 0.000000, 2.000000},
            {1.000000, 0.000000, 0.000000},  {0.309017, 0.951057, 0.000000},
            {-0.809017, 0.587785, 0.000000}, {-0.809017, -0.587785, 0.000000},
            {0.309017, -0.951057, 0.000000}};

    vector<Vector3i> ref_triangles = {
            {0, 3, 2}, {1, 2, 3}, {0, 4, 3}, {1, 3, 4}, {0, 5, 4},
            {1, 4, 5}, {0, 6, 5}, {1, 5, 6}, {0, 2, 6}, {1, 6, 2}};

    auto output_tm = geometry::TriangleMesh::CreateCone(1.0, 2.0, 5);

    ExpectEQ(ref_vertices, output_tm->vertices_);
    ExpectEQ(ref_triangles, output_tm->triangles_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshArrow) {
    vector<Vector3d> ref_vertices = {
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

    vector<Vector3i> ref_triangles = {
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

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshCoordinateFrame) {
    vector<Vector3d> ref_vertices = {
            {0.000000, 0.000000, 0.006000},   {0.000939, 0.000000, 0.005926},
            {0.000927, 0.000147, 0.005926},   {-0.000000, -0.002724, 0.005346},
            {0.000426, -0.002690, 0.005346},  {-0.000000, -0.003527, 0.004854},
            {-0.005346, 0.000000, 0.002724},  {-0.005280, -0.000836, 0.002724},
            {-0.005706, 0.000000, 0.001854},  {0.000000, 0.005926, -0.000939},
            {-0.000927, 0.005853, -0.000939}, {0.000000, 0.005706, -0.001854},
            {0.004243, 0.000000, -0.004243},  {0.004190, 0.000664, -0.004243},
            {0.003527, 0.000000, -0.004854},  {-0.000000, -0.001854, -0.005706},
            {0.000290, -0.001831, -0.005706}, {-0.000000, -0.000939, -0.005926},
            {0.000000, 0.080000, 0.003500},   {0.001082, 0.080000, 0.003329},
            {0.000000, 0.060000, 0.003500},   {-0.003500, 0.000000, 0.060000},
            {-0.003329, -0.001082, 0.060000}, {-0.003500, 0.000000, 0.040000}};

    vector<Vector3d> ref_vertex_normals = {
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

    vector<Vector3d> ref_vertex_colors = {
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

    vector<Vector3i> ref_triangles = {{0, 1, 2},    {5, 4, 3},    {8, 7, 6},
                                      {11, 10, 9},  {14, 13, 12}, {17, 16, 15},
                                      {20, 19, 18}, {23, 22, 21}};

    vector<Vector3d> ref_triangle_normals = {
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
    vector<size_t> indices;
    for (size_t i = 0; i < output_tm->triangles_.size(); i += stride) {
        indices.push_back(output_tm->triangles_[i](0, 0));
        indices.push_back(output_tm->triangles_[i](1, 0));
        indices.push_back(output_tm->triangles_[i](2, 0));
    }
    unique(indices.begin(), indices.end());
    sort(indices.begin(), indices.end());
    auto output = output_tm->SelectDownSample(indices);

    ExpectEQ(ref_vertices, output->vertices_);
    ExpectEQ(ref_vertex_normals, output->vertex_normals_);
    ExpectEQ(ref_vertex_colors, output->vertex_colors_);
    ExpectEQ(ref_triangles, output->triangles_);
    ExpectEQ(ref_triangle_normals, output->triangle_normals_);
}
