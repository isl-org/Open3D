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

#include "Open3D/Utility/Console.h"

static_assert(sizeof(float) == 4,
              "Unsupported platform: float must be 4 bytes");
static_assert(sizeof(double) == 8,
              "Unsupported platform: double must be 8 bytes");

namespace open3d {

enum class Dtype {
    undefined,
    float32,
    float64,
    int32,
    int64,
    uint8,
};

class DtypeUtil {
public:
    static size_t ByteSize(const Dtype &dtype) {
        size_t byte_size = 0;
        switch (dtype) {
            case Dtype::float32:
                byte_size = 4;
                break;
            case Dtype::float64:
                byte_size = 8;
                break;
            case Dtype::int32:
                byte_size = 4;
                break;
            case Dtype::int64:
                byte_size = 8;
                break;
            case Dtype::uint8:
                byte_size = 1;
                break;
            default:
                utility::LogFatal("Unsupported data type\n");
        }
        return byte_size;
    }

    template <typename T>
    static inline Dtype FromType() {
        utility::LogFatal("Unsupported data type\n");
        return Dtype::undefined;
    }

    static std::string ToString(const Dtype &dtype) {
        std::string str = "";
        switch (dtype) {
            case Dtype::float32:
                str = "float32";
                break;
            case Dtype::float64:
                str = "float64";
                break;
            case Dtype::int32:
                str = "int32";
                break;
            case Dtype::int64:
                str = "int64";
                break;
            case Dtype::uint8:
                str = "uint8";
                break;
            default:
                utility::LogFatal("Unsupported data type\n");
        }
        return str;
    }
};

template <>
inline Dtype DtypeUtil::FromType<float>() {
    return Dtype::float32;
}

template <>
inline Dtype DtypeUtil::FromType<double>() {
    return Dtype::float64;
}

template <>
inline Dtype DtypeUtil::FromType<int32_t>() {
    return Dtype::int32;
}

template <>
inline Dtype DtypeUtil::FromType<int64_t>() {
    return Dtype::int64;
}

template <>
inline Dtype DtypeUtil::FromType<uint8_t>() {
    return Dtype::uint8;
}

}  // namespace open3d
