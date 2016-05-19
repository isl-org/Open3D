// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "ViewParameters.h"

#include <Eigen/Dense>
#include <jsoncpp/include/json/json.h>
#include <Core/Utility/Console.h>

namespace three{

ViewParameters::Vector11d ViewParameters::ConvertToVector11d()
{
	ViewParameters::Vector11d v;
	v(0) = field_of_view_;
	v(1) = zoom_;
	v.block<3, 1>(2, 0) = lookat_;
	v.block<3, 1>(5, 0) = up_;
	v.block<3, 1>(8, 0) = front_;
	return v;
}

void ViewParameters::ConvertFromVector11d(const ViewParameters::Vector11d &v)
{
	field_of_view_ = v(0);
	zoom_ = v(1);
	lookat_ = v.block<3, 1>(2, 0);
	up_ = v.block<3, 1>(5, 0);
	front_ = v.block<3, 1>(8, 0);
}

bool ViewParameters::ConvertToJsonValue(Json::Value &value) const
{
	value["field_of_view"] = field_of_view_;
	value["zoom"] = zoom_;
	if (EigenVector3dToJsonArray(lookat_, value["lookat"]) == false ) {
		return false;
	}
	if (EigenVector3dToJsonArray(up_, value["up"]) == false ) {
		return false;
	}
	if (EigenVector3dToJsonArray(front_, value["front"]) == false ) {
		return false;
	}
	return true;
}

bool ViewParameters::ConvertFromJsonValue(const Json::Value &value)
{
	if (value.isObject() == false) {
		PrintWarning("ViewParameters read JSON failed: unsupported json format.\n");
		return false;		
	}
	field_of_view_ = value.get("field_of_view", 60.0).asDouble();
	zoom_ = value.get("zoom", 0.7).asDouble();
	if (EigenVector3dFromJsonArray(lookat_, value["lookat"]) == false) {
		PrintWarning("ViewParameters read JSON failed: wrong format.\n");
		return false;
	}
	if (EigenVector3dFromJsonArray(up_, value["up"]) == false) {
		PrintWarning("ViewParameters read JSON failed: wrong format.\n");
		return false;
	}
	if (EigenVector3dFromJsonArray(front_, value["front"]) == false) {
		PrintWarning("ViewParameters read JSON failed: wrong format.\n");
		return false;
	}
	return true;
}

}	// namespace three
