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

#include <Eigen/Dense>
#include <limits>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/utility/Console.h"

namespace open3d {

namespace {
using namespace geometry;

int CountValidDepthPixels(const Image &depth, int stride) {
    int num_valid_pixels = 0;
    for (int i = 0; i < depth.height_; i += stride) {
        for (int j = 0; j < depth.width_; j += stride) {
            const float *p = depth.PointerAt<float>(j, i);
            if (*p > 0) num_valid_pixels += 1;
        }
    }
    return num_valid_pixels;
}

std::shared_ptr<PointCloud> CreatePointCloudFromFloatDepthImage(
        const Image &depth,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4d &extrinsic,
        int stride,
        bool project_valid_depth_only) {
    auto pointcloud = std::make_shared<PointCloud>();
    Eigen::Matrix4d camera_pose = extrinsic.inverse();
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    int num_valid_pixels;
    if (!project_valid_depth_only) {
        num_valid_pixels =
                int(depth.height_ / stride) * int(depth.width_ / stride);
    } else {
        num_valid_pixels = CountValidDepthPixels(depth, stride);
    }
    pointcloud->points_.resize(num_valid_pixels);
    int cnt = 0;
    for (int i = 0; i < depth.height_; i += stride) {
        for (int j = 0; j < depth.width_; j += stride) {
            const float *p = depth.PointerAt<float>(j, i);
            if (*p > 0) {
                double z = (double)(*p);
                double x = (j - principal_point.first) * z / focal_length.first;
                double y =
                        (i - principal_point.second) * z / focal_length.second;
                Eigen::Vector4d point =
                        camera_pose * Eigen::Vector4d(x, y, z, 1.0);
                pointcloud->points_[cnt++] = point.block<3, 1>(0, 0);
            } else if (!project_valid_depth_only) {
                double z = std::numeric_limits<float>::quiet_NaN();
                double x = std::numeric_limits<float>::quiet_NaN();
                double y = std::numeric_limits<float>::quiet_NaN();
                pointcloud->points_[cnt++] = Eigen::Vector3d(x, y, z);
            }
        }
    }
    return pointcloud;
}

template <typename TC, int NC>
std::shared_ptr<PointCloud> CreatePointCloudFromRGBDImageT(
        const RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4d &extrinsic,
        bool project_valid_depth_only) {
    auto pointcloud = std::make_shared<PointCloud>();
    Eigen::Matrix4d camera_pose = extrinsic.inverse();
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    double scale = (sizeof(TC) == 1) ? 255.0 : 1.0;
    int num_valid_pixels;
    if (!project_valid_depth_only) {
        num_valid_pixels = image.depth_.height_ * image.depth_.width_;
    } else {
        num_valid_pixels = CountValidDepthPixels(image.depth_, 1);
    }
    pointcloud->points_.resize(num_valid_pixels);
    pointcloud->colors_.resize(num_valid_pixels);
    int cnt = 0;
    for (int i = 0; i < image.depth_.height_; i++) {
        float *p = (float *)(image.depth_.data_.data() +
                             i * image.depth_.BytesPerLine());
        TC *pc = (TC *)(image.color_.data_.data() +
                        i * image.color_.BytesPerLine());
        for (int j = 0; j < image.depth_.width_; j++, p++, pc += NC) {
            if (*p > 0) {
                double z = (double)(*p);
                double x = (j - principal_point.first) * z / focal_length.first;
                double y =
                        (i - principal_point.second) * z / focal_length.second;
                Eigen::Vector4d point =
                        camera_pose * Eigen::Vector4d(x, y, z, 1.0);
                pointcloud->points_[cnt] = point.block<3, 1>(0, 0);
                pointcloud->colors_[cnt++] =
                        Eigen::Vector3d(pc[0], pc[(NC - 1) / 2], pc[NC - 1]) /
                        scale;
            } else if (!project_valid_depth_only) {
                double z = std::numeric_limits<float>::quiet_NaN();
                double x = std::numeric_limits<float>::quiet_NaN();
                double y = std::numeric_limits<float>::quiet_NaN();
                pointcloud->points_[cnt] = Eigen::Vector3d(x, y, z);
                pointcloud->colors_[cnt++] =
                        Eigen::Vector3d(std::numeric_limits<TC>::quiet_NaN(),
                                        std::numeric_limits<TC>::quiet_NaN(),
                                        std::numeric_limits<TC>::quiet_NaN());
            }
        }
    }
    return pointcloud;
}

}  // unnamed namespace

namespace geometry {
std::shared_ptr<PointCloud> PointCloud::CreateFromDepthImage(
        const Image &depth,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4d &extrinsic /* = Eigen::Matrix4d::Identity()*/,
        double depth_scale /* = 1000.0*/,
        double depth_trunc /* = 1000.0*/,
        int stride /* = 1*/,
        bool project_valid_depth_only) {
    if (depth.num_of_channels_ == 1) {
        if (depth.bytes_per_channel_ == 2) {
            auto float_depth =
                    depth.ConvertDepthToFloatImage(depth_scale, depth_trunc);
            return CreatePointCloudFromFloatDepthImage(
                    *float_depth, intrinsic, extrinsic, stride,
                    project_valid_depth_only);
        } else if (depth.bytes_per_channel_ == 4) {
            return CreatePointCloudFromFloatDepthImage(
                    depth, intrinsic, extrinsic, stride,
                    project_valid_depth_only);
        }
    }
    utility::LogError(
            "[CreatePointCloudFromDepthImage] Unsupported image format.");
    return std::make_shared<PointCloud>();
}

std::shared_ptr<PointCloud> PointCloud::CreateFromRGBDImage(
        const RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4d &extrinsic /* = Eigen::Matrix4d::Identity()*/,
        bool project_valid_depth_only) {
    if (image.depth_.num_of_channels_ == 1 &&
        image.depth_.bytes_per_channel_ == 4) {
        if (image.color_.bytes_per_channel_ == 1 &&
            image.color_.num_of_channels_ == 3) {
            return CreatePointCloudFromRGBDImageT<uint8_t, 3>(
                    image, intrinsic, extrinsic, project_valid_depth_only);
        } else if (image.color_.bytes_per_channel_ == 1 &&
                   image.color_.num_of_channels_ == 4) {
            return CreatePointCloudFromRGBDImageT<uint8_t, 4>(
                    image, intrinsic, extrinsic, project_valid_depth_only);
        } else if (image.color_.bytes_per_channel_ == 4 &&
                   image.color_.num_of_channels_ == 1) {
            return CreatePointCloudFromRGBDImageT<float, 1>(
                    image, intrinsic, extrinsic, project_valid_depth_only);
        }
    }
    utility::LogError(
            "[CreatePointCloudFromRGBDImage] Unsupported image format.");
    return std::make_shared<PointCloud>();
}

std::shared_ptr<PointCloud> PointCloud::CreateFromVoxelGrid(
        const VoxelGrid &voxel_grid) {
    auto output = std::make_shared<PointCloud>();
    output->points_.resize(voxel_grid.voxels_.size());
    bool has_colors = voxel_grid.HasColors();
    if (has_colors) {
        output->colors_.resize(voxel_grid.voxels_.size());
    }
    size_t vidx = 0;
    for (auto &it : voxel_grid.voxels_) {
        const geometry::Voxel voxel = it.second;
        output->points_[vidx] =
                voxel_grid.GetVoxelCenterCoordinate(voxel.grid_index_);
        if (has_colors) {
            output->colors_[vidx] = voxel.color_;
        }
        vidx++;
    }
    return output;
}

}  // namespace geometry
}  // namespace open3d
