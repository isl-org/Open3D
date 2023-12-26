// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/camera/PinholeCameraParameters.h"

#include <json/json.h>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace camera {
PinholeCameraParameters::PinholeCameraParameters() {}

PinholeCameraParameters::~PinholeCameraParameters() {}

bool PinholeCameraParameters::ConvertToJsonValue(Json::Value &value) const {
    Json::Value trajectory_array;
    value["class_name"] = "PinholeCameraParameters";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    if (!EigenMatrix4dToJsonArray(extrinsic_, value["extrinsic"])) {
        return false;
    }
    if (!intrinsic_.ConvertToJsonValue(value["intrinsic"])) {
        return false;
    }
    return true;
}

bool PinholeCameraParameters::ConvertFromJsonValue(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PinholeCameraParameters" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (!intrinsic_.ConvertFromJsonValue(value["intrinsic"])) {
        return false;
    }
    if (!EigenMatrix4dFromJsonArray(extrinsic_, value["extrinsic"])) {
        return false;
    }
    return true;
}
}  // namespace camera
}  // namespace open3d
