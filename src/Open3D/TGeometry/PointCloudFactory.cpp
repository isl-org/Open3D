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
#include <type_traits>
#include "Open3D/TGeometry/PointCloud.h"

namespace open3d {
namespace tgeometry {

template <typename T>
inline Tensor FromTemplatedVectorXd(const Eigen::VectorXd& vector,
                                    int64_t N,
                                    Dtype dtype,
                                    Device device) {
    auto vals = std::vector<T>(N);
    for (int64_t i = 0; i < N; ++i) {
        vals[i] = vector(i);
    }
    return Tensor(vals, SizeVector({N}), dtype, device);
}

Tensor FromVectorXd(const Eigen::VectorXd& vector, Dtype dtype, Device device) {
    int64_t N = vector.size();
    if (dtype == Dtype::Float64) {
        return FromTemplatedVectorXd<double>(vector, N, dtype, device);
    } else if (dtype == Dtype::Float32) {
        return FromTemplatedVectorXd<float>(vector, N, dtype, device);
    } else {
        utility::LogError("Unsupported dtype for FromVectorXd.");
    }
}

inline Eigen::Vector3d FromTensor3d(const Tensor& tensor) {
    auto dtype = tensor.GetDtype();
    if (dtype == Dtype::Float64) {
        return Eigen::Vector3d(tensor[0].Item<double>(),
                               tensor[1].Item<double>(),
                               tensor[2].Item<double>());
    } else if (dtype == Dtype::Float32) {
        return Eigen::Vector3d(tensor[0].Item<float>(), tensor[1].Item<float>(),
                               tensor[2].Item<float>());
    } else {
        utility::LogError("Unsupported dtype for FromTensor3d.");
    }
}

tgeometry::PointCloud PointCloud::FromLegacyPointCloud(
        const geometry::PointCloud& pcd_legacy) {
    tgeometry::PointCloud pcd;

    if (!pcd_legacy.HasPoints()) {
        utility::LogWarning("Create from empty geometry::PointCloud");
        return pcd;
    }

    int64_t N = pcd_legacy.points_.size();
    pcd.point_dict_["points"].Resize(N);

    auto pts_tensor = pcd.point_dict_["points"].AsTensor();
    auto dtype = pts_tensor.GetDtype();
    auto device = pts_tensor.GetDevice();

    for (int64_t i = 0; i < N; ++i) {
        pts_tensor[i] = FromVectorXd(pcd_legacy.points_[i], dtype, device);
    }

    if (pcd_legacy.HasNormals()) {
        int64_t N = pcd_legacy.normals_.size();
        pcd.point_dict_["normals"] = TensorList({3}, dtype, device, N);

        auto ns_tensor = pcd.point_dict_["normals"].AsTensor();
        for (int64_t i = 0; i < N; ++i) {
            ns_tensor[i] = FromVectorXd(pcd_legacy.normals_[i], dtype, device);
        }
    }

    if (pcd_legacy.HasColors()) {
        int64_t N = pcd_legacy.colors_.size();
        pcd.point_dict_["colors"] = TensorList({3}, dtype, device, N);

        auto cs_tensor = pcd.point_dict_["colors"].AsTensor();
        for (int64_t i = 0; i < N; ++i) {
            cs_tensor[i] = FromVectorXd(pcd_legacy.colors_[i], dtype, device);
        }
    }

    return pcd;
}

std::shared_ptr<geometry::PointCloud> PointCloud::ToLegacyPointCloud(
        const tgeometry::PointCloud& pcd) {
    auto pcd_legacy = std::make_shared<geometry::PointCloud>();

    if (!pcd.HasPoints()) {
        utility::LogWarning("Create from empty tgeometry::PointCloud");
        return pcd_legacy;
    }

    auto pts_tensorlist = pcd.point_dict_.find("points")->second;
    auto pts_tensor = pts_tensorlist.AsTensor();
    int64_t N = pts_tensorlist.GetSize();

    pcd_legacy->points_.resize(N);
    for (int64_t i = 0; i < N; ++i) {
        pcd_legacy->points_[i] = FromTensor3d(pts_tensor[i]);
    }

    if (pcd.HasColors()) {
        auto cs_tensor = pcd.point_dict_.find("colors")->second.AsTensor();
        pcd_legacy->colors_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            pcd_legacy->colors_[i] = FromTensor3d(cs_tensor[i]);
        }
    }

    if (pcd.HasNormals()) {
        auto ns_tensor = pcd.point_dict_.find("normals")->second.AsTensor();
        pcd_legacy->normals_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            pcd_legacy->normals_[i] = FromTensor3d(ns_tensor[i]);
        }
    }

    return pcd_legacy;
}
}  // namespace tgeometry
}  // namespace open3d
