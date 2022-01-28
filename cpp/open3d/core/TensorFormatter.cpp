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

#include "open3d/core/TensorFormatter.h"

#include <limits>
#include <sstream>
#include <string>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

// Global singleton class
class PrintOptionsState {
public:
    static PrintOptionsState& GetInstance() {
        static PrintOptionsState instance;
        return instance;
    }
    PrintOptionsState(const PrintOptionsState&) = delete;
    PrintOptionsState& operator=(const PrintOptionsState&) = delete;

    void SetPrintOptions(const PrintOptions& print_options) {
        print_options_ = print_options;
    }
    const PrintOptions& GetPrintOptions() const { return print_options_; }

private:
    PrintOptionsState();
    PrintOptions print_options_;
};

void SetPrintOptions(utility::optional<int> precision,
                     utility::optional<int> threshold,
                     utility::optional<int> edgeitems,
                     utility::optional<int> linewidth,
                     utility::optional<std::string> profile,
                     utility::optional<bool> sci_mode) {
    PrintOptions print_options;
    if (profile.has_value()) {
        if (profile.value() == "default") {
            print_options.precision_ = 4;
            print_options.threshold_ = 1000;
            print_options.edgeitems_ = 3;
            print_options.linewidth_ = 80;
        } else if (profile.value() == "short") {
            print_options.precision_ = 2;
            print_options.threshold_ = 1000;
            print_options.edgeitems_ = 2;
            print_options.linewidth_ = 80;
        } else if (profile.value() == "full") {
            print_options.precision_ = 4;
            print_options.threshold_ = std::numeric_limits<int>::max();
            print_options.edgeitems_ = 3;
            print_options.linewidth_ = 80;
        } else {
            utility::LogWarning("Unknown profile: {}", profile.value());
        }
    }
    if (precision.has_value()) {
        print_options.precision_ = precision.value();
    }
    if (threshold.has_value()) {
        print_options.threshold_ = threshold.value();
    }
    if (edgeitems.has_value()) {
        print_options.edgeitems_ = edgeitems.value();
    }
    if (linewidth.has_value()) {
        print_options.linewidth_ = linewidth.value();
    }
    print_options.sci_mode_ = sci_mode;

    PrintOptionsState::GetInstance().SetPrintOptions(print_options);
}

PrintOptions GetPrintOptions() {
    return PrintOptionsState::GetInstance().GetPrintOptions();
}

static std::string ScalarPtrToString(const void* ptr, const Dtype& dtype) {
    std::string str = "";
    if (dtype == core::Bool) {
        str = *static_cast<const unsigned char*>(ptr) ? "True" : "False";
    } else if (dtype.IsObject()) {
        str = fmt::format("{}", fmt::ptr(ptr));
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
            str = fmt::format("{}", *static_cast<const scalar_t*>(ptr));
        });
    }
    return str;
}

static std::string ToString(const Tensor& tensor,
                            bool with_suffix = true,
                            const std::string& indent = "") {
    std::ostringstream rc;
    if (tensor.GetDevice().GetType() == Device::DeviceType::CUDA ||
        !tensor.IsContiguous()) {
        Tensor host_contiguous_tensor = tensor.Contiguous().To(Device("CPU:0"));
        rc << ToString(host_contiguous_tensor, false, "");
    } else {
        if (tensor.GetShape().NumElements() == 0) {
            rc << indent;
            rc << "0-element Tensor";
        } else if (tensor.GetShape().size() == 0) {
            rc << indent;
            rc << ScalarPtrToString(tensor.GetDataPtr(), tensor.GetDtype());
        } else if (tensor.GetShape().size() == 1) {
            const char* ptr = static_cast<const char*>(tensor.GetDataPtr());
            rc << "[";
            std::string delim = "";
            int64_t element_byte_size = tensor.GetDtype().ByteSize();
            for (int64_t i = 0; i < tensor.GetShape().NumElements(); ++i) {
                rc << delim << ScalarPtrToString(ptr, tensor.GetDtype());
                delim = " ";
                ptr += element_byte_size;
            }
            rc << "]";
        } else {
            rc << "[";
            std::string delim = "";
            std::string child_indent = "";
            for (int64_t i = 0; i < tensor.GetShape()[0]; ++i) {
                rc << delim << child_indent
                   << ToString(tensor[i], false, indent + " ");
                delim = ",\n";
                child_indent = indent + " ";
            }
            rc << "]";
        }
    }
    if (with_suffix) {
        rc << fmt::format("\nTensor[shape={}, stride={}, {}, {}, {}]",
                          tensor.GetShape().ToString(),
                          tensor.GetStrides().ToString(),
                          tensor.GetDtype().ToString(),
                          tensor.GetDevice().ToString(), tensor.GetDataPtr());
    }
    return rc.str();
}

std::string FormatTensor(const Tensor& tensor, bool with_suffix) {
    return ToString(tensor, with_suffix);
}

}  // namespace core
}  // namespace open3d
