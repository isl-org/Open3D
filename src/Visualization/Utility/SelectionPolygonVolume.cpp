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

#include "SelectionPolygonVolume.h"

#include <jsoncpp/include/json/json.h>
#include <Core/Utility/Console.h>

namespace three{

bool SelectionPolygonVolume::ConvertToJsonValue(Json::Value &value) const
{
	Json::Value polygon_array;
	for (const auto &point : bounding_polygon_) {
		Json::Value point_object;
		if (EigenVector3dToJsonArray(point, point_object) == false) {
			return false;
		}
		polygon_array.append(point_object);
	}
	value["class_name"] = "SelectionPolygonVolume";
	value["version_major"] = 1;
	value["version_minor"] = 0;
	value["bounding_polygon"] = polygon_array;
	value["orthogonal_axis"] = orthogonal_axis_;
	return true;
}

bool SelectionPolygonVolume::ConvertFromJsonValue(const Json::Value &value)
{
	if (value.isObject() == false) {
		PrintWarning("SelectionPolygonVolume read JSON failed: unsupported json format.\n");
		return false;		
	}
	if (value.get("class_name", "").asString() != "SelectionPolygonVolume" ||
			value.get("version_major", 1).asInt() != 1 ||
			value.get("version_minor", 0).asInt() != 0) {
		PrintWarning("SelectionPolygonVolume read JSON failed: unsupported json format.\n");
		return false;
	}
	orthogonal_axis_ = value.get("orthogonal_axis", "").asString();
	const Json::Value &polygon_array = value["bounding_polygon"];
	if (polygon_array.size() == 0) {
		PrintWarning("SelectionPolygonVolume read JSON failed: empty trajectory.\n");
		return false;
	}
	bounding_polygon_.resize(polygon_array.size());
	for (int i = 0; i < (int)polygon_array.size(); i++) {
		const Json::Value &point_object = polygon_array[i];
		if (EigenVector3dFromJsonArray(bounding_polygon_[i], point_object) ==
				false) {
			return false;
		}
	}
	return true;
}

}	// namespace three
