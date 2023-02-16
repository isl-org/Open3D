// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/TensorCheck.h"

#include <string>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace tensor_check {

void AssertTensorDtype_(const char* file,
                        int line,
                        const char* function,
                        const Tensor& tensor,
                        const Dtype& dtype) {
    if (tensor.GetDtype() == dtype) {
        return;
    }
    std::string error_message =
            fmt::format("Tensor has dtype {}, but is expected to have {}.",
                        tensor.GetDtype().ToString(), dtype.ToString());
    utility::Logger::LogError_(file, line, function, error_message.c_str());
}

void AssertTensorDtypes_(const char* file,
                         int line,
                         const char* function,
                         const Tensor& tensor,
                         const std::vector<Dtype>& dtypes) {
    for (auto& it : dtypes) {
        if (tensor.GetDtype() == it) {
            return;
        }
    }

    std::vector<std::string> dtype_strings;
    for (const Dtype& dtype : dtypes) {
        dtype_strings.push_back(dtype.ToString());
    }
    std::string error_message = fmt::format(
            "Tensor has dtype {}, but is expected to have dtype among {{{}}}.",
            tensor.GetDtype().ToString(), utility::JoinStrings(dtype_strings));
    utility::Logger::LogError_(file, line, function, error_message.c_str());
}

void AssertTensorDevice_(const char* file,
                         int line,
                         const char* function,
                         const Tensor& tensor,
                         const Device& device) {
    if (tensor.GetDevice() == device) {
        return;
    }
    std::string error_message =
            fmt::format("Tensor has device {}, but is expected to have {}.",
                        tensor.GetDevice().ToString(), device.ToString());
    utility::Logger::LogError_(file, line, function, error_message.c_str());
}

void AssertTensorShape_(const char* file,
                        int line,
                        const char* function,
                        const Tensor& tensor,
                        const DynamicSizeVector& shape) {
    if (shape.IsDynamic()) {
        if (tensor.GetShape().IsCompatible(shape)) {
            return;
        }
        std::string error_message = fmt::format(
                "Tensor has shape {}, but is expected to have compatible with "
                "{}.",
                tensor.GetShape().ToString(), shape.ToString());
        utility::Logger::LogError_(file, line, function, error_message.c_str());
    } else {
        SizeVector static_shape = shape.ToSizeVector();
        if (tensor.GetShape() == static_shape) {
            return;
        }
        std::string error_message = fmt::format(
                "Tensor has shape {}, but is expected to have {}.",
                tensor.GetShape().ToString(), static_shape.ToString());
        utility::Logger::LogError_(file, line, function, error_message.c_str());
    }
}

}  // namespace tensor_check
}  // namespace core
}  // namespace open3d
