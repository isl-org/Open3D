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
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/RobustKernel.h"

#include <algorithm>
#include <cmath>

namespace {
double inline square(double x) { return x * x; }
}  // namespace

namespace open3d {
namespace pipelines {
namespace registration {

double L2Loss::Weight(double /*residual*/) const { return 1.0; }

double L1Loss::Weight(double residual) const {
    return 1.0 / std::abs(residual);
}

double HuberLoss::Weight(double residual) const {
    const double e = std::abs(residual);
    return k_ / std::max(e, k_);
}

double CauchyLoss::Weight(double residual) const {
    return 1.0 / (1 + square(residual / k_));
}

double GMLoss::Weight(double residual) const {
    return k_ / square(k_ + square(residual));
}

double TukeyLoss::Weight(double residual) const {
    const double e = std::abs(residual);
    return square(1.0 - square(std::min(1.0, e / k_)));
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
