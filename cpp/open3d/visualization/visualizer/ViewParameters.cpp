// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/ViewParameters.h"

#include <json/json.h>

#include <Eigen/Dense>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {

ViewParameters::Vector17d ViewParameters::ConvertToVector17d() {
    ViewParameters::Vector17d v;
    v(0) = field_of_view_;
    v(1) = zoom_;
    v.block<3, 1>(2, 0) = lookat_;
    v.block<3, 1>(5, 0) = up_;
    v.block<3, 1>(8, 0) = front_;
    v.block<3, 1>(11, 0) = boundingbox_min_;
    v.block<3, 1>(14, 0) = boundingbox_max_;
    return v;
}

void ViewParameters::ConvertFromVector17d(const ViewParameters::Vector17d &v) {
    field_of_view_ = v(0);
    zoom_ = v(1);
    lookat_ = v.block<3, 1>(2, 0);
    up_ = v.block<3, 1>(5, 0);
    front_ = v.block<3, 1>(8, 0);
    boundingbox_min_ = v.block<3, 1>(11, 0);
    boundingbox_max_ = v.block<3, 1>(14, 0);
}

bool ViewParameters::ConvertToJsonValue(Json::Value &value) const {
    value["field_of_view"] = field_of_view_;
    value["zoom"] = zoom_;
    if (!EigenVector3dToJsonArray(lookat_, value["lookat"])) {
        return false;
    }
    if (!EigenVector3dToJsonArray(up_, value["up"])) {
        return false;
    }
    if (!EigenVector3dToJsonArray(front_, value["front"])) {
        return false;
    }
    if (!EigenVector3dToJsonArray(boundingbox_min_, value["boundingbox_min"])) {
        return false;
    }
    if (!EigenVector3dToJsonArray(boundingbox_max_, value["boundingbox_max"])) {
        return false;
    }
    return true;
}

bool ViewParameters::ConvertFromJsonValue(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "ViewParameters read JSON failed: unsupported json format.");
        return false;
    }
    field_of_view_ = value.get("field_of_view", 60.0).asDouble();
    zoom_ = value.get("zoom", 0.7).asDouble();
    if (!EigenVector3dFromJsonArray(lookat_, value["lookat"])) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    if (!EigenVector3dFromJsonArray(up_, value["up"])) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    if (!EigenVector3dFromJsonArray(front_, value["front"])) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    if (!EigenVector3dFromJsonArray(boundingbox_min_,
                                    value["boundingbox_min"])) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    if (!EigenVector3dFromJsonArray(boundingbox_max_,
                                    value["boundingbox_max"])) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    return true;
}

}  // namespace visualization
}  // namespace open3d
