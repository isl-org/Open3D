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

#include "Open3D/Visualization/Visualizer/ViewParameters.h"

#include <json/json.h>
#include <Eigen/Dense>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace visualization {

ViewParameters::Vector17d ViewParameters::ConvertToVector17d() {
    ViewParameters::Vector17d v;
    v(0) = field_of_view_;
    v(1) = zoom_;
    v.block<3, 1>(2, 0) = lookat_;
    v.block<3, 1>(5, 0) = up_;
    v.block<3, 1>(8, 0) = front_;
    v.block<3, 1>(11, 0) = boundingbox_min_;
    v.block<3, 1>(14, 0) = boundingbox_max_;
    return v;
}

void ViewParameters::ConvertFromVector17d(const ViewParameters::Vector17d &v) {
    field_of_view_ = v(0);
    zoom_ = v(1);
    lookat_ = v.block<3, 1>(2, 0);
    up_ = v.block<3, 1>(5, 0);
    front_ = v.block<3, 1>(8, 0);
    boundingbox_min_ = v.block<3, 1>(11, 0);
    boundingbox_max_ = v.block<3, 1>(14, 0);
}

bool ViewParameters::ConvertToJsonValue(Json::Value &value) const {
    value["field_of_view"] = field_of_view_;
    value["zoom"] = zoom_;
    if (EigenVector3dToJsonArray(lookat_, value["lookat"]) == false) {
        return false;
    }
    if (EigenVector3dToJsonArray(up_, value["up"]) == false) {
        return false;
    }
    if (EigenVector3dToJsonArray(front_, value["front"]) == false) {
        return false;
    }
    if (EigenVector3dToJsonArray(boundingbox_min_, value["boundingbox_min"]) ==
        false) {
        return false;
    }
    if (EigenVector3dToJsonArray(boundingbox_max_, value["boundingbox_max"]) ==
        false) {
        return false;
    }
    return true;
}

bool ViewParameters::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "ViewParameters read JSON failed: unsupported json format.\n");
        return false;
    }
    field_of_view_ = value.get("field_of_view", 60.0).asDouble();
    zoom_ = value.get("zoom", 0.7).asDouble();
    if (EigenVector3dFromJsonArray(lookat_, value["lookat"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.\n");
        return false;
    }
    if (EigenVector3dFromJsonArray(up_, value["up"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.\n");
        return false;
    }
    if (EigenVector3dFromJsonArray(front_, value["front"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.\n");
        return false;
    }
    if (EigenVector3dFromJsonArray(boundingbox_min_,
                                   value["boundingbox_min"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.\n");
        return false;
    }
    if (EigenVector3dFromJsonArray(boundingbox_max_,
                                   value["boundingbox_max"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.\n");
        return false;
    }
    return true;
}

}  // namespace visualization
}  // namespace open3d
