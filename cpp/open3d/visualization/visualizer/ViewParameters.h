// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "open3d/utility/IJsonConvertible.h"

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
