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

#include <cmath>

namespace open3d {
namespace pipelines {
namespace registration {

double L2Loss::Weight(double /*residual*/) const { return 1.0; }

double L1Loss::Weight(double residual) const {
    return 1.0 / std::abs(residual);
}

double HuberLoss::Weight(double residual) const {
    const double e = std::abs(residual);
    if (e > k_) {
        return k_ / e;
    }
    return 1.0;
}

double CauchyLoss::Weight(double residual) const {
    return 1.0 / (1 + std::pow(residual / k_, 2.0));
}

double GMLoss::Weight(double residual) const {
    return k_ / std::pow(k_ + std::pow(residual, 2.0), 2.0);
}

double TukeyLoss::Weight(double residual) const {
    const double e = std::abs(residual);
    if (e > k_) {
        return 0.0;
    }
    return std::pow((1.0 - std::pow(e / k_, 2.0)), 2.0);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
