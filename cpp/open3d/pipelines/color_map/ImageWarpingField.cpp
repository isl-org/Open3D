// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/pipelines/color_map/ImageWarpingField.h"

#include <json/json.h>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace color_map {

ImageWarpingField::ImageWarpingField() { InitializeWarpingFields(0, 0, 0); }

ImageWarpingField::ImageWarpingField(int width,
                                     int height,
                                     int number_of_vertical_anchors) {
    InitializeWarpingFields(width, height, number_of_vertical_anchors);
}

void ImageWarpingField::InitializeWarpingFields(
        int width, int height, int number_of_vertical_anchors) {
    anchor_h_ = number_of_vertical_anchors;
    anchor_step_ = double(height) / (anchor_h_ - 1);
    anchor_w_ = int(std::ceil(double(width) / anchor_step_) + 1);
    flow_ = Eigen::VectorXd(anchor_w_ * anchor_h_ * 2);
    for (int i = 0; i <= (anchor_w_ - 1); i++) {
        for (int j = 0; j <= (anchor_h_ - 1); j++) {
            int baseidx = (i + j * anchor_w_) * 2;
            flow_(baseidx) = i * anchor_step_;
            flow_(baseidx + 1) = j * anchor_step_;
        }
    }
}

Eigen::Vector2d ImageWarpingField::QueryFlow(int i, int j) const {
    int baseidx = (i + j * anchor_w_) * 2;
    // exceptional case: queried anchor index is out of pre-defined space
    if (baseidx < 0 || baseidx >= anchor_w_ * anchor_h_ * 2)
        return Eigen::Vector2d(0.0, 0.0);
    else
        return Eigen::Vector2d(flow_(baseidx), flow_(baseidx + 1));
}

Eigen::Vector2d ImageWarpingField::GetImageWarpingField(double u,
                                                        double v) const {
    int i = (int)(u / anchor_step_);
    int j = (int)(v / anchor_step_);
    double p = (u - i * anchor_step_) / anchor_step_;
    double q = (v - j * anchor_step_) / anchor_step_;
    return (1 - p) * (1 - q) * QueryFlow(i, j) +
           (1 - p) * (q)*QueryFlow(i, j + 1) +
           (p) * (1 - q) * QueryFlow(i + 1, j) +
           (p) * (q)*QueryFlow(i + 1, j + 1);
}

bool ImageWarpingField::ConvertToJsonValue(Json::Value &value) const {
    value["class_name"] = "ImageWarpingField";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    value["anchor_w"] = anchor_w_;
    value["anchor_h"] = anchor_h_;
    Json::Value flow_array;
    for (int i = 0; i < anchor_w_ * anchor_h_ * 2; i++) {
        flow_array.append(flow_[i]);
    }
    value["flow"] = flow_array;
    return true;
}

bool ImageWarpingField::ConvertFromJsonValue(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "ImageWarpingField read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "ImageWarpingField" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "ImageWarpingField read JSON failed: unsupported json "
                "format.");
        return false;
    }
    anchor_w_ = value.get("anchor_w", 1).asInt();
    anchor_h_ = value.get("anchor_h", 1).asInt();

    const Json::Value flow_array = value["flow"];
    if (flow_array.size() == 0 ||
        int(flow_array.size()) != (anchor_w_ * anchor_h_ * 2)) {
        utility::LogWarning(
                "ImageWarpingField read JSON failed: invalid flow.");
        return false;
    }
    flow_.resize(anchor_w_ * anchor_h_ * 2, 1);
    for (int i = 0; i < anchor_w_ * anchor_h_ * 2; i++) {
        flow_(i) = flow_array[i].asDouble();
    }
    return true;
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
