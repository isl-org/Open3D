// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/camera/PinholeCameraTrajectory.h"

#include <json/json.h>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace camera {

PinholeCameraTrajectory::PinholeCameraTrajectory() {}

PinholeCameraTrajectory::~PinholeCameraTrajectory() {}

bool PinholeCameraTrajectory::ConvertToJsonValue(Json::Value &value) const {
    value["class_name"] = "PinholeCameraTrajectory";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    Json::Value parameters_array;
    for (const auto &parameter : parameters_) {
        Json::Value parameter_value;
        parameter.ConvertToJsonValue(parameter_value);
        parameters_array.append(parameter_value);
    }
    value["parameters"] = parameters_array;

    return true;
}

bool PinholeCameraTrajectory::ConvertFromJsonValue(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "PinholeCameraTrajectory read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PinholeCameraTrajectory" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PinholeCameraTrajectory read JSON failed: unsupported json "
                "format.");
        return false;
    }

    const Json::Value parameter_array = value["parameters"];

    if (parameter_array.size() == 0) {
        utility::LogWarning(
                "PinholeCameraTrajectory read JSON failed: empty "
                "trajectory.");
        return false;
    }
    parameters_.resize(parameter_array.size());
    for (size_t i = 0; i < parameter_array.size(); i++) {
        const Json::Value &status_object = parameter_array[int(i)];
        if (!parameters_[i].intrinsic_.ConvertFromJsonValue(
                    status_object["intrinsic"])) {
            return false;
        }
        if (!EigenMatrix4dFromJsonArray(parameters_[i].extrinsic_,
                                        status_object["extrinsic"])) {
            return false;
        }
    }
    return true;
}
}  // namespace camera
}  // namespace open3d
