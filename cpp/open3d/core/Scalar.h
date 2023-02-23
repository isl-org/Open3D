// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "open3d/core/Dtype.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

/// Scalar is a stores one of {double, int64, bool}. Typically Scalar is used to
/// accept C++ scalar arguments of different types via implicit conversion
/// constructor. Doing so can avoid the need for templates.
class Scalar {
public:
    enum class ScalarType { Double, Int64, Bool };

    Scalar(float v) {
        scalar_type_ = ScalarType::Double;
        value_.d = static_cast<double>(v);
    }
    Scalar(double v) {
        scalar_type_ = ScalarType::Double;
        value_.d = static_cast<double>(v);
    }
    Scalar(int8_t v) {
        scalar_type_ = ScalarType::Int64;
        value_.i = static_cast<int64_t>(v);
    }
    Scalar(int16_t v) {
        scalar_type_ = ScalarType::Int64;
        value_.i = static_cast<int64_t>(v);
    }
    Scalar(int32_t v) {
        scalar_type_ = ScalarType::Int64;
        value_.i = static_cast<int64_t>(v);
    }
    Scalar(int64_t v) {
        scalar_type_ = ScalarType::Int64;
        value_.i = static_cast<int64_t>(v);
    }

    // This constructor is required to ensure long input support where int64_t
    // is not equal to long (e.g. mac os where int64_t is long long).
    // The template argument with enable_if ensures that this constructor is
    // enabled only when int64_t is not equal to long.
    // Ref: https://en.cppreference.com/w/cpp/types/enable_if
    template <typename T = int64_t>
    Scalar(long v,
           typename std::enable_if<!std::is_same<T, long>::value>::type* = 0) {
        scalar_type_ = ScalarType::Int64;
        value_.i = static_cast<int64_t>(v);
    }
    Scalar(uint8_t v) {
        scalar_type_ = ScalarType::Int64;
        value_.i = static_cast<int64_t>(v);
    }
    Scalar(uint16_t v) {
        scalar_type_ = ScalarType::Int64;
        value_.i = static_cast<int64_t>(v);
    }
    Scalar(uint32_t v) {
        scalar_type_ = ScalarType::Int64;
        value_.i = static_cast<int64_t>(v);
    }
    Scalar(uint64_t v) {
        scalar_type_ = ScalarType::Int64;
        // Conversion uint64_t -> int64_t is undefined behaviour until C++20.
        // Compilers optimize this to a single cast.
        if (v <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            value_.i = static_cast<int64_t>(v);
        } else {
            // Safe conversion to two's complement:
            // - Compute x = uint_max - v such that x <= int_max
            // - Safely cast x from unsigned to signed
            // - Map x to y such that casting y to signed leads y = v
            value_.i = -static_cast<int64_t>(
                               std::numeric_limits<uint64_t>::max() - v) -
                       1;
        }
    }
    Scalar(bool v) {
        scalar_type_ = ScalarType::Bool;
        value_.b = static_cast<bool>(v);
    }

    bool IsDouble() const { return scalar_type_ == ScalarType::Double; }
    bool IsInt64() const { return scalar_type_ == ScalarType::Int64; }
    bool IsBool() const { return scalar_type_ == ScalarType::Bool; }

    /// Returns double value from Scalar. Only works when scalar_type_ is
    /// ScalarType::Double.
    double GetDouble() const {
        if (!IsDouble()) {
            utility::LogError("Scalar is not a ScalarType:Double type.");
        }
        return value_.d;
    }
    /// Returns int64 value from Scalar. Only works when scalar_type_ is
    /// ScalarType::Int64.
    int64_t GetInt64() const {
        if (!IsInt64()) {
            utility::LogError("Scalar is not a ScalarType:Int64 type.");
        }
        return value_.i;
    }
    /// Returns bool value from Scalar. Only works when scalar_type_ is
    /// ScalarType::Bool.
    bool GetBool() const {
        if (!IsBool()) {
            utility::LogError("Scalar is not a ScalarType:Bool type.");
        }
        return value_.b;
    }

    /// To<T>() does not check for scalar type and overflows.
    template <typename T>
    T To() const {
        if (scalar_type_ == ScalarType::Double) {
            return static_cast<T>(value_.d);
        } else if (scalar_type_ == ScalarType::Int64) {
            return static_cast<T>(value_.i);
        } else if (scalar_type_ == ScalarType::Bool) {
            return static_cast<T>(value_.b);
        } else {
            utility::LogError("To: ScalarType not supported.");
        }
    }

    void AssertSameScalarType(Scalar other,
                              const std::string& error_msg) const {
        if (scalar_type_ != other.scalar_type_) {
            if (error_msg.empty()) {
                utility::LogError("Scalar mode {} are not the same as {}.",
                                  ToString(), other.ToString());
            } else {
                utility::LogError("Scalar mode {} are not the same as {}: {}",
                                  ToString(), other.ToString(), error_msg);
            }
        }
    }

    std::string ToString() const {
        std::string scalar_type_str;
        std::string value_str;
        if (scalar_type_ == ScalarType::Double) {
            scalar_type_str = "Double";
            value_str = std::to_string(value_.d);
        } else if (scalar_type_ == ScalarType::Int64) {
            scalar_type_str = "Int64";
            value_str = std::to_string(value_.i);
        } else if (scalar_type_ == ScalarType::Bool) {
            scalar_type_str = "Bool";
            value_str = value_.b ? "true" : "false";
        } else {
            utility::LogError("ScalarTypeToString: ScalarType not supported.");
        }
        return scalar_type_str + ":" + value_str;
    }

    template <typename T>
    bool Equal(T value) const {
        if (scalar_type_ == ScalarType::Double) {
            return value_.d == value;
        } else if (scalar_type_ == ScalarType::Int64) {
            return value_.i == value;
        } else if (scalar_type_ == ScalarType::Bool) {
            return false;  // Boolean does not equal to non-boolean values.
        } else {
            utility::LogError("Equals: ScalarType not supported.");
        }
    }

    bool Equal(bool value) const {
        return scalar_type_ == ScalarType::Bool && value_.b == value;
    }

    bool Equal(Scalar other) const {
        if (other.scalar_type_ == ScalarType::Double) {
            return Equal(other.GetDouble());
        } else if (other.scalar_type_ == ScalarType::Int64) {
            return Equal(other.GetInt64());
        } else if (other.scalar_type_ == ScalarType::Bool) {
            return scalar_type_ == ScalarType::Bool &&
                   value_.b == other.value_.b;
        } else {
            utility::LogError("Equals: ScalarType not supported.");
        }
    }

private:
    ScalarType scalar_type_;
    union value_t {
        double d;
        int64_t i;
        bool b;
    } value_;
};

}  // namespace core
}  // namespace open3d
