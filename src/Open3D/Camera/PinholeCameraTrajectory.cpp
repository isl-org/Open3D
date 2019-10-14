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

#include "Open3D/Camera/PinholeCameraTrajectory.h"

#include <json/json.h>

#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include "Open3D/Utility/Console.h"

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
    if (value.isObject() == false) {
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
        if (parameters_[i].intrinsic_.ConvertFromJsonValue(
                    status_object["intrinsic"]) == false) {
            return false;
        }
        if (EigenMatrix4dFromJsonArray(parameters_[i].extrinsic_,
                                       status_object["extrinsic"]) == false) {
            return false;
        }
    }
    return true;
}
}  // namespace camera
}  // namespace open3d
