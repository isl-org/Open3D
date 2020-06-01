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

#include "Open3D/TGeometry/PointCloud.h"
#include "Open3D/Core/TensorList.h"

#include "Core/CoreTest.h"
#include "TestUtility/UnitTest.h"

namespace open3d {
namespace unit_test {

class PointCloudPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(PointCloud,
                         PointCloudPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class PointCloudPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        PointCloud,
        PointCloudPermuteDevicePairs,
        testing::ValuesIn(PointCloudPermuteDevicePairs::TestCases()));

using namespace tgeometry;

TEST_P(PointCloudPermuteDevices, DefaultConstructor) {
    PointCloud pc;

    // Inherited from Geometry3D.
    EXPECT_EQ(pc.GetGeometryType(), Geometry::GeometryType::PointCloud);
    EXPECT_EQ(pc.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(pc.IsEmpty());
    EXPECT_FALSE(pc.HasPoints());
}

TEST_P(PointCloudPermuteDevices, GetMinBound_GetMaxBound_GetCenter) {
    Device device = GetParam();
    PointCloud pc(Dtype::Float32, device);

    TensorList& points = pc.point_dict_["points"];
    points.PushBack(
            Tensor(std::vector<float>{1, 2, 3}, {3}, Dtype::Float32, device));
    points.PushBack(
            Tensor(std::vector<float>{4, 5, 6}, {3}, Dtype::Float32, device));

    EXPECT_FALSE(pc.IsEmpty());
    EXPECT_TRUE(pc.HasPoints());
    EXPECT_EQ(pc.GetMinBound().ToFlatVector<float>(),
              std::vector<float>({1, 2, 3}));
    EXPECT_EQ(pc.GetMaxBound().ToFlatVector<float>(),
              std::vector<float>({4, 5, 6}));
    EXPECT_EQ(pc.GetCenter().ToFlatVector<float>(),
              std::vector<float>({2.5, 3.5, 4.5}));
}

TEST_P(PointCloudPermuteDevices, Scale) {
    Device device = GetParam();
    PointCloud pc;
    TensorList& points = pc.point_dict_["points"];
    points = TensorList(Tensor(std::vector<float>{0, 0, 0, 1, 1, 1, 2, 2, 2},
                               {3, 3}, Dtype::Float32, device));
    Tensor center(std::vector<float>{1, 1, 1}, {3}, Dtype::Float32, device);
    float scale = 4;
    pc.Scale(scale, center);
    EXPECT_EQ(points.AsTensor().ToFlatVector<float>(),
              std::vector<float>({-3, -3, -3, 1, 1, 1, 5, 5, 5}));
}

}  // namespace unit_test
}  // namespace open3d
