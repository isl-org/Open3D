// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <fmt/format.h>

#include <Eigen/Core>

#include "open3d/utility/Eigen.h"

/// @cond
namespace Json {
class Value;
}  // namespace Json
/// @endcond

namespace open3d {
namespace utility {

/// \brief Parse string and convert to Json::value. Throws exception if the
/// conversion is invalid.
///
/// \param json_str String containing json value.
/// \return A Json object.
Json::Value StringToJson(const std::string &json_str);

/// \brief Serialize a Json::Value to a string.
///
/// \param json The Json::Value object to be converted.
/// \return A string containing the json value.
std::string JsonToString(const Json::Value &json);

/// Class IJsonConvertible defines the behavior of a class that can convert
/// itself to/from a json::Value.
class IJsonConvertible {
public:
    virtual ~IJsonConvertible() {}

public:
    virtual bool ConvertToJsonValue(Json::Value &value) const = 0;
    virtual bool ConvertFromJsonValue(const Json::Value &value) = 0;

    /// Convert to a styled string representation of JSON data for display
    virtual std::string ToString() const;

public:
    static bool EigenVector3dFromJsonArray(Eigen::Vector3d &vec,
                                           const Json::Value &value);
    static bool EigenVector3dToJsonArray(const Eigen::Vector3d &vec,
                                         Json::Value &value);
    static bool EigenVector4dFromJsonArray(Eigen::Vector4d &vec,
                                           const Json::Value &value);
    static bool EigenVector4dToJsonArray(const Eigen::Vector4d &vec,
                                         Json::Value &value);
    static bool EigenMatrix3dFromJsonArray(Eigen::Matrix3d &mat,
                                           const Json::Value &value);
    static bool EigenMatrix3dToJsonArray(const Eigen::Matrix3d &mat,
                                         Json::Value &value);
    static bool EigenMatrix4dFromJsonArray(Eigen::Matrix4d &mat,
                                           const Json::Value &value);
    static bool EigenMatrix4dToJsonArray(const Eigen::Matrix4d &mat,
                                         Json::Value &value);
    static bool EigenMatrix4dFromJsonArray(Eigen::Matrix4d_u &mat,
                                           const Json::Value &value);
    static bool EigenMatrix4dToJsonArray(const Eigen::Matrix4d_u &mat,
                                         Json::Value &value);
    static bool EigenMatrix6dFromJsonArray(Eigen::Matrix6d &mat,
                                           const Json::Value &value);
    static bool EigenMatrix6dToJsonArray(const Eigen::Matrix6d &mat,
                                         Json::Value &value);
    static bool EigenMatrix6dFromJsonArray(Eigen::Matrix6d_u &mat,
                                           const Json::Value &value);
    static bool EigenMatrix6dToJsonArray(const Eigen::Matrix6d_u &mat,
                                         Json::Value &value);
};

}  // namespace utility
}  // namespace open3d

namespace fmt {
template <>
struct formatter<Json::Value> {
    template <typename FormatContext>
    auto format(const Json::Value &value, FormatContext &ctx)
            -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{}", open3d::utility::JsonToString(value));
    }

    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
};

}  // namespace fmt
