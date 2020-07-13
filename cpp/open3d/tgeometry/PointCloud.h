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

#pragma once

#include <Eigen/Core>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/tgeometry/Geometry3D.h"

namespace open3d {
namespace tgeometry {

/// \class PointCloud
/// \brief A PointCloud contains point coordinates, and optionally point colors
/// and point normals.
class PointCloud : public Geometry3D {
public:
    PointCloud(core::Dtype dtype = core::Dtype::Float32,
               core::Device device = core::Device("CPU:0"))
        : Geometry3D(Geometry::GeometryType::PointCloud),
          dtype_(dtype),
          device_(device) {
        point_dict_.emplace("points", core::TensorList({3}, dtype_, device_));
    }

    /// Construct from default points
    /// points_tensor: (N, 3)
    PointCloud(const core::Tensor &points_tensor);

    /// Construct from points and various other properties
    PointCloud(const std::unordered_map<std::string, core::TensorList>
                       &point_dict);

    ~PointCloud() override {}

    core::TensorList &operator[](const std::string &key);

    void SyncPushBack(
            const std::unordered_map<std::string, core::Tensor> &point_struct);

    PointCloud &Clear() override;

    bool IsEmpty() const override;

    core::Tensor GetMinBound() const override;

    core::Tensor GetMaxBound() const override;

    core::Tensor GetCenter() const override;

    PointCloud &Transform(const core::Tensor &transformation) override;

    PointCloud &Translate(const core::Tensor &translation,
                          bool relative = true) override;

    PointCloud &Scale(double scale, const core::Tensor &center) override;

    PointCloud &Rotate(const core::Tensor &R,
                       const core::Tensor &center) override;

public:
    bool HasPoints() const {
        return point_dict_.find("points") != point_dict_.end() &&
               point_dict_.at("points").GetSize() > 0;
    }

    bool HasColors() const {
        return point_dict_.find("colors") != point_dict_.end() &&
               point_dict_.at("colors").GetSize() > 0;
    }

    bool HasNormals() const {
        return point_dict_.find("normals") != point_dict_.end() &&
               point_dict_.at("normals").GetSize() > 0;
    }

public:
    std::unordered_map<std::string, core::TensorList> point_dict_;

public:
    // Usage:
    // std::shared_ptr<geometry::PointCloud> pcd_legacy =
    //         io::CreatePointCloudFromFile(filename);
    // tgeometry::PointCloud pcd =
    //         tgeometry::PointCloud::FromLegacyPointCloud(*pcd_legacy);
    // geometry::PointCloud pcd_legacy_back =
    //         tgeometry::PointCloud::ToLegacyPointCloud(pcd);
    static tgeometry::PointCloud FromLegacyPointCloud(
            const geometry::PointCloud &pcd_legacy,
            core::Dtype dtype = core::Dtype::Float32,
            core::Device device = core::Device("CPU:0"));

    static geometry::PointCloud ToLegacyPointCloud(
            const tgeometry::PointCloud &pcd);

protected:
    core::Dtype dtype_ = core::Dtype::Float32;
    core::Device device_ = core::Device("CPU:0");
};

}  // namespace tgeometry
}  // namespace open3d
