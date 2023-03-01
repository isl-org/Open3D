// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <thread>

#include "open3d/Open3D.h"

namespace open3d {

class AlignmentSession : public utility::IJsonConvertible {
public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    std::shared_ptr<geometry::PointCloud>
            source_ptr_;  // Original source pointcloud
    std::shared_ptr<geometry::PointCloud>
            target_ptr_;                  // Original target pointcloud
    std::vector<size_t> source_indices_;  // Manually annotated point indices
    std::vector<size_t> target_indices_;  // Manually annotated point indices
    Eigen::Matrix4d_u transformation_;    // Current alignment result
    double voxel_size_ = -1.0;
    double max_correspondence_distance_ = -1.0;
    bool with_scaling_ = true;
};

}  // namespace open3d
