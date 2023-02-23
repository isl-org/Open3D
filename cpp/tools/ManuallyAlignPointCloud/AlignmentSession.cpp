// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tools/ManuallyAlignPointCloud/AlignmentSession.h"

#include <json/json.h>

namespace open3d {

bool AlignmentSession::ConvertToJsonValue(Json::Value &value) const {
    value["class_name"] = "AlignmentSession";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    Json::Value source_array;
    for (const auto &si : source_indices_) {
        source_array.append((int)si);
    }
    value["source_indices"] = source_array;
    Json::Value target_array;
    for (const auto &ti : target_indices_) {
        target_array.append((int)ti);
    }
    value["target_indices"] = target_array;
    if (!EigenMatrix4dToJsonArray(transformation_, value["transformation"])) {
        return false;
    }
    value["voxel_size"] = voxel_size_;
    value["max_correspondence_distance"] = max_correspondence_distance_;
    value["with_scaling"] = with_scaling_;
    return true;
}

bool AlignmentSession::ConvertFromJsonValue(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "AlignmentSession read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "AlignmentSession" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "AlignmentSession read JSON failed: unsupported json "
                "format.");
        return false;
    }
    const auto &source_array = value["source_indices"];
    source_indices_.resize(source_array.size());
    for (int i = 0; i < (int)source_array.size(); i++) {
        source_indices_[i] = (size_t)source_array[i].asInt();
    }
    const auto &target_array = value["target_indices"];
    target_indices_.resize(target_array.size());
    for (int i = 0; i < (int)target_array.size(); i++) {
        target_indices_[i] = (size_t)target_array[i].asInt();
    }
    if (!EigenMatrix4dFromJsonArray(transformation_, value["transformation"])) {
        return false;
    }
    voxel_size_ = value["voxel_size"].asDouble();
    max_correspondence_distance_ =
            value["max_correspondence_distance"].asDouble();
    with_scaling_ = value["with_scaling"].asBool();
    return true;
}

}  // namespace open3d
