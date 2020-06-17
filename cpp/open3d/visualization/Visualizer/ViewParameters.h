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

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "Open3D/Utility/IJsonConvertible.h"

namespace open3d {
namespace visualization {

class ViewParameters : public utility::IJsonConvertible {
public:
    typedef Eigen::Matrix<double, 17, 4, Eigen::RowMajor> Matrix17x4d;
    typedef Eigen::Matrix<double, 17, 1> Vector17d;
    typedef Eigen::aligned_allocator<Matrix17x4d> Matrix17x4d_allocator;

public:
    ViewParameters()
        : field_of_view_(0),
          zoom_(0),
          lookat_(0, 0, 0),
          up_(0, 0, 0),
          front_(0, 0, 0),
          boundingbox_min_(0, 0, 0),
          boundingbox_max_(0, 0, 0) {}
    ~ViewParameters() override {}

public:
    Vector17d ConvertToVector17d();
    void ConvertFromVector17d(const Vector17d &v);
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    double field_of_view_;
    double zoom_;
    Eigen::Vector3d lookat_;
    Eigen::Vector3d up_;
    Eigen::Vector3d front_;
    Eigen::Vector3d boundingbox_min_;
    Eigen::Vector3d boundingbox_max_;
};

}  // namespace visualization
}  // namespace open3d
