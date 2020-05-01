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

#include "string"

#include "Open3D/Core/Dispatch.h"
#include "Open3D/Utility/Console.h"

static_assert(sizeof(float) == 4,
              "Unsupported platform: float must be 4 bytes");
static_assert(sizeof(double) == 8,
              "Unsupported platform: double must be 8 bytes");
static_assert(sizeof(int) == 4, "Unsupported platform: int must be 4 bytes");
static_assert(sizeof(int32_t) == 4,
              "Unsupported platform: int32_t must be 4 bytes");
static_assert(sizeof(int64_t) == 8,
              "Unsupported platform: int64_t must be 8 bytes");
static_assert(sizeof(uint8_t) == 1,
              "Unsupported platform: uint8_t must be 1 byte");
static_assert(sizeof(bool) == 1, "Unsupported platform: bool must be 1 byte");

namespace open3d {

enum class Dtype {
    Undefined,  // Dtype for uninitialized Tensor
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
    Bool,
};

class DtypeUtil {
public:
    static int64_t ByteSize(const Dtype &dtype) {
        int64_t byte_size = 0;
        switch (dtype) {
            case Dtype::Float32:
                byte_size = 4;
                break;
            case Dtype::Float64:
                byte_size = 8;
                break;
            case Dtype::Int32:
                byte_size = 4;
                break;
            case Dtype::Int64:
                byte_size = 8;
                break;
            case Dtype::UInt8:
                byte_size = 1;
                break;
            case Dtype::Bool:
                byte_size = 1;
                break;
            default:
                utility::LogError("Unsupported data type");
        }
        return byte_size;
    }

    /// Convert from C++ types to Dtype. Known types are explicitly specialized,
    /// e.g. DtypeUtil::FromType<float>(). Unsupported type will result in an
    /// exception.
    template <typename T>
    static inline Dtype FromType() {
        utility::LogError("Unsupported data type");
        return Dtype::Undefined;
    }

    static std::string ToString(const Dtype &dtype) {
        std::string str = "";
        switch (dtype) {
            case Dtype::Undefined:
                str = "Undefined";
                break;
            case Dtype::Float32:
                str = "Float32";
                break;
            case Dtype::Float64:
                str = "Float64";
                break;
            case Dtype::Int32:
                str = "Int32";
                break;
            case Dtype::Int64:
                str = "Int64";
                break;
            case Dtype::UInt8:
                str = "UInt8";
                break;
            case Dtype::Bool:
                str = "Bool";
                break;
            default:
                utility::LogError("Unsupported data type");
        }
        return str;
    }
};

template <>
inline Dtype DtypeUtil::FromType<float>() {
    return Dtype::Float32;
}

template <>
inline Dtype DtypeUtil::FromType<double>() {
    return Dtype::Float64;
}

template <>
inline Dtype DtypeUtil::FromType<int32_t>() {
    return Dtype::Int32;
}

template <>
inline Dtype DtypeUtil::FromType<int64_t>() {
    return Dtype::Int64;
}

template <>
inline Dtype DtypeUtil::FromType<uint8_t>() {
    return Dtype::UInt8;
}

template <>
inline Dtype DtypeUtil::FromType<bool>() {
    return Dtype::Bool;
}

}  // namespace open3d
