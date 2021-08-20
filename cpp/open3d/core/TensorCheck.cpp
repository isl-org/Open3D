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

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace tensor_check {

void _AssertTensorDtype(const char* file,
                        int line,
                        const char* function,
                        const Tensor& tensor,
                        const Dtype& dtype) {
    if (tensor.GetDtype() == dtype) {
        return;
    }
    std::string error_message =
            fmt::format("Tensor has dtype {}, but is expected to be {}.",
                        tensor.GetDtype().ToString(), dtype.ToString());
    utility::Logger::_LogError(file, line, function, error_message.c_str());
}

void _AssertTensorDevice(const char* file,
                         int line,
                         const char* function,
                         const Tensor& tensor,
                         const Device& device) {
    if (tensor.GetDevice() == device) {
        return;
    }
    std::string error_message =
            fmt::format("Tensor has device {}, but is expected to be {}.",
                        tensor.GetDevice().ToString(), device.ToString());
    utility::Logger::_LogError(file, line, function, error_message.c_str());
}

void _AssertTensorShape(const char* file,
                        int line,
                        const char* function,
                        const Tensor& tensor,
                        const DynamicSizeVector& shape) {
    if (shape.IsDynamic()) {
        if (tensor.GetShape().IsCompatible(shape)) {
            return;
        }
        std::string error_message = fmt::format(
                "Tensor has shape {}, but is expected to be compatible with "
                "{}.",
                tensor.GetShape().ToString(), shape.ToString());
        utility::Logger::_LogError(file, line, function, error_message.c_str());
    } else {
        SizeVector static_shape = shape.ToSizeVector();
        if (tensor.GetShape() == static_shape) {
            return;
        }
        std::string error_message = fmt::format(
                "Tensor has shape {}, but is expected to be {}.",
                tensor.GetShape().ToString(), static_shape.ToString());
        utility::Logger::_LogError(file, line, function, error_message.c_str());
    }
}

}  // namespace tensor_check
}  // namespace core
}  // namespace open3d
