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
                                    Device device = Device("CPU:0")) {
    auto vals = std::vector<T>(N);
    for (int64_t i = 0; i < N; ++i) {
        vals[i] = vector(i);
    }
    return Tensor(vals, SizeVector({N}), dtype, device);
}

Tensor FromVectorXd(const Eigen::VectorXd& vector,
                    Dtype dtype,
                    Device device = Device("CPU:0")) {
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
    Dtype dtype = tensor.GetDtype();
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
        const geometry::PointCloud& pcd_legacy, Dtype dtype, Device device) {
    tgeometry::PointCloud pcd(dtype, device);

    if (pcd_legacy.HasPoints()) {
        int64_t N = pcd_legacy.points_.size();
        Tensor pts_tensor({N, 3}, dtype);
        for (int64_t i = 0; i < N; ++i) {
            pts_tensor[i] = FromVectorXd(pcd_legacy.points_[i], dtype);
        }
        pcd.point_dict_["points"] =
                TensorList::FromTensor(pts_tensor.Copy(device));
    } else {
        utility::LogWarning(
                "Create from empty geometry::PointCloud, returning an empty "
                "tgeometry::PointCloud");
        return pcd;
    }

    if (pcd_legacy.HasColors()) {
        int64_t N = pcd_legacy.colors_.size();
        Tensor cs_tensor({N, 3}, dtype);
        for (int64_t i = 0; i < N; ++i) {
            cs_tensor[i] = FromVectorXd(pcd_legacy.colors_[i], dtype);
        }
        pcd.point_dict_["colors"] =
                TensorList::FromTensor(cs_tensor.Copy(device));
    }

    if (pcd_legacy.HasNormals()) {
        int64_t N = pcd_legacy.normals_.size();
        Tensor ns_tensor({N, 3}, dtype);
        for (int64_t i = 0; i < N; ++i) {
            ns_tensor[i] = FromVectorXd(pcd_legacy.normals_[i], dtype);
        }
        pcd.point_dict_["normals"] =
                TensorList::FromTensor(ns_tensor.Copy(device));
    }

    return pcd;
}

geometry::PointCloud PointCloud::ToLegacyPointCloud(
        const tgeometry::PointCloud& pcd) {
    geometry::PointCloud pcd_legacy;

    Device host = Device("CPU:0");

    if (pcd.HasPoints()) {
        Tensor pts_tensor =
                pcd.point_dict_.find("points")->second.AsTensor().Copy(host);
        int64_t N = pts_tensor.GetShape()[0];
        pcd_legacy.points_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            pcd_legacy.points_[i] = FromTensor3d(pts_tensor[i]);
        }
    }

    if (pcd.HasColors()) {
        Tensor cs_tensor =
                pcd.point_dict_.find("colors")->second.AsTensor().Copy(host);
        int64_t N = cs_tensor.GetShape()[0];
        pcd_legacy.colors_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            pcd_legacy.colors_[i] = FromTensor3d(cs_tensor[i]);
        }
    }

    if (pcd.HasNormals()) {
        Tensor ns_tensor =
                pcd.point_dict_.find("normals")->second.AsTensor().Copy(host);
        int64_t N = ns_tensor.GetShape()[0];
        pcd_legacy.normals_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            pcd_legacy.normals_[i] = FromTensor3d(ns_tensor[i]);
        }
    }

    return pcd_legacy;
}
}  // namespace tgeometry
}  // namespace open3d
