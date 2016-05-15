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

#include "IJsonConvertible.h"

#include <jsoncpp/include/json/json.h>

namespace three{

bool IJsonConvertible::EigenVector3dFromJsonArray(Eigen::Vector3d &vec,
		const Json::Value &value)
{
	if (value.size() != 3) {
		return false;
	} else {
		vec(0) = value[0].asDouble();
		vec(1) = value[1].asDouble();
		vec(2) = value[2].asDouble();
		return true;
	}
}

bool IJsonConvertible::EigenVector3dToJsonArray(const Eigen::Vector3d &vec,
		Json::Value &value)
{
	value.clear();
	value.append(vec(0));
	value.append(vec(1));
	value.append(vec(2));
	return true;
}

}	// namespace three
