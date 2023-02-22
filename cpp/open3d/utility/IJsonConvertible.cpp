// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/IJsonConvertible.h"

#include <json/json.h>

#include <string>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

Json::Value StringToJson(const std::string &json_str) {
    Json::Value json;
    std::string err;
    Json::CharReaderBuilder builder;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(json_str.c_str(), json_str.c_str() + json_str.length(),
                       &json, &err)) {
        utility::LogError("Failed to parse string to json, error: {}", err);
    }
    return json;
}

std::string JsonToString(const Json::Value &json) {
    return Json::writeString(Json::StreamWriterBuilder(), json);
}

std::string IJsonConvertible::ToString() const {
    Json::Value val;
    ConvertToJsonValue(val);
    return val.toStyledString();
}

bool IJsonConvertible::EigenVector3dFromJsonArray(Eigen::Vector3d &vec,
                                                  const Json::Value &value) {
    if (value.size() != 3) {
        return false;
    } else {
        vec(0) = value[0].asDouble();
        vec(1) = value[1].asDouble();
        vec(2) = value[2].asDouble();
        return true;
    }
}

bool IJsonConvertible::EigenVector3dToJsonArray(const Eigen::Vector3d &vec,
                                                Json::Value &value) {
    value.clear();
    value.append(vec(0));
    value.append(vec(1));
    value.append(vec(2));
    return true;
}

bool IJsonConvertible::EigenVector4dFromJsonArray(Eigen::Vector4d &vec,
                                                  const Json::Value &value) {
    if (value.size() != 4) {
        return false;
    } else {
        vec(0) = value[0].asDouble();
        vec(1) = value[1].asDouble();
        vec(2) = value[2].asDouble();
        vec(3) = value[3].asDouble();
        return true;
    }
}

bool IJsonConvertible::EigenVector4dToJsonArray(const Eigen::Vector4d &vec,
                                                Json::Value &value) {
    value.clear();
    value.append(vec(0));
    value.append(vec(1));
    value.append(vec(2));
    value.append(vec(3));
    return true;
}

bool IJsonConvertible::EigenMatrix3dFromJsonArray(Eigen::Matrix3d &mat,
                                                  const Json::Value &value) {
    if (value.size() != 9) {
        return false;
    } else {
        for (int i = 0; i < 9; i++) {
            mat.coeffRef(i) = value[i].asDouble();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix3dToJsonArray(const Eigen::Matrix3d &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 9; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

bool IJsonConvertible::EigenMatrix4dFromJsonArray(Eigen::Matrix4d &mat,
                                                  const Json::Value &value) {
    if (value.size() != 16) {
        return false;
    } else {
        for (int i = 0; i < 16; i++) {
            mat.coeffRef(i) = value[i].asDouble();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix4dToJsonArray(const Eigen::Matrix4d &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 16; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

bool IJsonConvertible::EigenMatrix4dFromJsonArray(Eigen::Matrix4d_u &mat,
                                                  const Json::Value &value) {
    if (value.size() != 16) {
        return false;
    } else {
        for (int i = 0; i < 16; i++) {
            mat.coeffRef(i) = value[i].asDouble();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix4dToJsonArray(const Eigen::Matrix4d_u &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 16; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

bool IJsonConvertible::EigenMatrix6dFromJsonArray(Eigen::Matrix6d &mat,
                                                  const Json::Value &value) {
    if (value.size() != 36) {
        return false;
    } else {
        for (int i = 0; i < 36; i++) {
            mat.coeffRef(i) = value[i].asDouble();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix6dToJsonArray(const Eigen::Matrix6d &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 36; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

bool IJsonConvertible::EigenMatrix6dFromJsonArray(Eigen::Matrix6d_u &mat,
                                                  const Json::Value &value) {
    if (value.size() != 36) {
        return false;
    } else {
        for (int i = 0; i < 36; i++) {
            mat.coeffRef(i) = value[i].asDouble();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix6dToJsonArray(const Eigen::Matrix6d_u &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 36; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

}  // namespace utility
}  // namespace open3d
