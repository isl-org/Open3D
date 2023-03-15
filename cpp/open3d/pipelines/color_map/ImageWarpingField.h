// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/utility/IJsonConvertible.h"

namespace open3d {
namespace pipelines {
namespace color_map {

class ImageWarpingField : public utility::IJsonConvertible {
public:
    ImageWarpingField();
    ImageWarpingField(int width, int height, int number_of_vertical_anchors);
    void InitializeWarpingFields(int width,
                                 int height,
                                 int number_of_vertical_anchors);
    Eigen::Vector2d QueryFlow(int i, int j) const;
    Eigen::Vector2d GetImageWarpingField(double u, double v) const;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    Eigen::VectorXd flow_;
    int anchor_w_;
    int anchor_h_;
    double anchor_step_;
};

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
