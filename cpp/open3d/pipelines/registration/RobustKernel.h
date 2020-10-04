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

#pragma once

namespace open3d {
namespace pipelines {
namespace registration {

enum class RobustKernelType {
    Unspecified = 0,
    L2 = 1,
    L1 = 2,
    Huber = 3,
    Tukey = 4,
};

class RobustKernel {
public:
    virtual ~RobustKernel() = default;
    virtual double Weight(double /*residual*/) const { return -1.0; };
    virtual inline RobustKernelType GetRobustKernelType() const {
        return RobustKernelType::Unspecified;
    }
};

class L2Loss : public RobustKernel {
public:
    double Weight(double residual) const override;
    inline RobustKernelType GetRobustKernelType() const override {
        return RobustKernelType::L2;
    }
};

class L1Loss : public RobustKernel {
public:
    double Weight(double residual) const override;
    inline RobustKernelType GetRobustKernelType() const override {
        return RobustKernelType::L1;
    }
};

class HuberLoss : public RobustKernel {
public:
    explicit HuberLoss(double k) : k_(k) {}

public:
    double Weight(double residual) const override;
    inline RobustKernelType GetRobustKernelType() const override {
        return RobustKernelType::Huber;
    }

public:
    double k_;
};

class TukeyLoss : public RobustKernel {
public:
    explicit TukeyLoss(double k) : k_(k) {}

public:
    double Weight(double residual) const override;
    inline RobustKernelType GetRobustKernelType() const override {
        return RobustKernelType::Tukey;
    }

public:
    double k_;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
