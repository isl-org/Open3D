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

#include "UnitTest.h"
#include "Core/Geometry/PointCloud.h"

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, Constructor)
{
    open3d::PointCloud pointCloud;

    // inherited from Geometry2D
    EXPECT_EQ(open3d::Geometry::GeometryType::PointCloud, pointCloud.GetGeometryType());
    EXPECT_EQ(3, pointCloud.Dimension());

    // public member variables
    EXPECT_EQ(0, pointCloud.points_.size());
    EXPECT_EQ(0, pointCloud.normals_.size());
    EXPECT_EQ(0, pointCloud.colors_.size());

    // public members
    EXPECT_TRUE(pointCloud.IsEmpty());
    EXPECT_EQ(Eigen::Vector3d(0, 0, 0), pointCloud.GetMinBound());
    EXPECT_EQ(Eigen::Vector3d(0, 0, 0), pointCloud.GetMaxBound());
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_Destructor)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_MemberData)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_Clear)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_IsEmpty)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_GetMinBound)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_GetMaxBound)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_Transform)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_HasPoints)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_HasNormals)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_HasColors)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_NormalizeNormals)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_PaintUniformColor)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromFile)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromDepthImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromRGBDImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_SelectDownSample)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_VoxelDownSample)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_UniformDownSample)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CropPointCloud)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_EstimateNormals)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_KDTreeSearchParamKNN)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_OrientNormalsToAlignWithDirection)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_OrientNormalsTowardsCameraLocation)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ComputePointCloudToPointCloudDistance)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ComputePointCloudMeanAndCovariance)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ComputePointCloudMahalanobisDistance)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ComputePointCloudNearestNeighborDistance)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromFloatDepthImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_PointerAt)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromRGBDImageT)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloudFactory, DISABLED_CreatePointCloudFromDepthImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ConvertDepthToFloatImage)
{
    NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(PointCloudFactory, DISABLED_CreatePointCloudFromRGBDImage)
{
    NotImplemented();
}
