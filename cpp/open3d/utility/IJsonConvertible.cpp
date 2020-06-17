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

#include "Open3D/Utility/IJsonConvertible.h"

#include <json/json.h>

namespace open3d {
namespace utility {

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
