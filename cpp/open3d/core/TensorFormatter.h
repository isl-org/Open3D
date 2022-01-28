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

// TODO: this location is just for debugging. Move it to somewhere else later.

#pragma once

#include <string>

#include "open3d/core/Tensor.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace core {

struct PrintOptions {
    int precision_ = 4;
    int threshold_ = 1000;
    int edgeitems_ = 3;
    int linewidth_ = 80;
    utility::optional<bool> sci_mode_ = utility::nullopt;
};

/// \brief Set options for printing tensors.
void SetPrintOptions(utility::optional<int> precision,
                     utility::optional<int> threshold,
                     utility::optional<int> edgeitems,
                     utility::optional<int> linewidth,
                     utility::optional<std::string> profile,
                     utility::optional<bool> sci_mode);

PrintOptions GetPrintOptions();

std::string FormatTensor(const Tensor& tensor, bool with_suffix = true);

}  // namespace core
}  // namespace open3d
