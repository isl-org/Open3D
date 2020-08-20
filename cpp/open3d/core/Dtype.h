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

#pragma once

#include <string>

#include "open3d/core/Dispatch.h"
#include "open3d/utility/Console.h"

static_assert(sizeof(float) == 4,
              "Unsupported platform: float must be 4 bytes.");
static_assert(sizeof(double) == 8,
              "Unsupported platform: double must be 8 bytes.");
static_assert(sizeof(int) == 4, "Unsupported platform: int must be 4 bytes.");
static_assert(sizeof(int32_t) == 4,
              "Unsupported platform: int32_t must be 4 bytes.");
static_assert(sizeof(int64_t) == 8,
              "Unsupported platform: int64_t must be 8 bytes.");
static_assert(sizeof(uint8_t) == 1,
              "Unsupported platform: uint8_t must be 1 byte.");
static_assert(sizeof(uint16_t) == 2,
              "Unsupported platform: uint16_t must be 2 bytes.");
static_assert(sizeof(bool) == 1, "Unsupported platform: bool must be 1 byte.");

namespace open3d {
namespace core {

class Dtype {
public:
    static const Dtype Undefined;
    static const Dtype Float32;
    static const Dtype Float64;
    static const Dtype Int32;
    static const Dtype Int64;
    static const Dtype UInt8;
    static const Dtype UInt16;
    static const Dtype Bool;

public:
    enum class DtypeCode {
        Undefined,
        Bool,  // Needed to distinguish bool from uint8_t.
        Int,
        UInt,
        Float,
        Object,
    };

    Dtype() : Dtype(DtypeCode::Undefined, 1, "Undefined") {}

    Dtype(DtypeCode dtype_code, int64_t byte_size, const std::string name)
        : dtype_code_(dtype_code), byte_size_(byte_size), name_(name) {
        (void)dtype_code_;
        (void)byte_size_;
        (void)name_;
    }

    /// Convert from C++ types to Dtype. Known types are explicitly specialized,
    /// e.g. DtypeUtil::FromType<float>(). Unsupported type will result in an
    /// exception.
    template <typename T>
    static inline const Dtype FromType() {
        utility::LogError("Unsupported data type");
        return Dtype::Undefined;
    }

    int64_t ByteSize() const { return byte_size_; }

    std::string ToString() const { return name_; }

    bool operator==(const Dtype &other) const {
        return dtype_code_ == other.dtype_code_ &&
               byte_size_ == other.byte_size_ && name_ == other.name_;
    }

    bool operator!=(const Dtype &other) const { return !(*this == other); }

private:
    DtypeCode dtype_code_;
    int64_t byte_size_;
    std::string name_;
};

template <>
inline const Dtype Dtype::FromType<float>() {
    return Dtype::Float32;
}

template <>
inline const Dtype Dtype::FromType<double>() {
    return Dtype::Float64;
}

template <>
inline const Dtype Dtype::FromType<int32_t>() {
    return Dtype::Int32;
}

template <>
inline const Dtype Dtype::FromType<int64_t>() {
    return Dtype::Int64;
}

template <>
inline const Dtype Dtype::FromType<uint8_t>() {
    return Dtype::UInt8;
}

template <>
inline const Dtype Dtype::FromType<uint16_t>() {
    return Dtype::UInt16;
}

template <>
inline const Dtype Dtype::FromType<bool>() {
    return Dtype::Bool;
}

class DtypeUtil {
public:
    static int64_t ByteSize(const Dtype &dtype) { return dtype.ByteSize(); }

    template <typename T>
    static const Dtype FromType() {
        return Dtype::FromType<T>();
    }

    static std::string ToString(const Dtype &dtype) { return dtype.ToString(); }
};

}  // namespace core
}  // namespace open3d
