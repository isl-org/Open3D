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

#include "PinholeCameraTrajectory.h"

#include <json/json.h>
#include <Core/Utility/Console.h>

namespace three{

PinholeCameraTrajectory::PinholeCameraTrajectory()
{
}

PinholeCameraTrajectory::~PinholeCameraTrajectory()
{
}

bool PinholeCameraTrajectory::ConvertToJsonValue(Json::Value &value) const
{
	Json::Value trajectory_array;
	for (const auto &status : extrinsic_) {
		Json::Value status_object;
		if (EigenMatrix4dToJsonArray(status, status_object) == false) {
			return false;
		}
		trajectory_array.append(status_object);
	}
	value["class_name"] = "PinholeCameraTrajectory";
	value["version_major"] = 1;
	value["version_minor"] = 0;
	value["extrinsic"] = trajectory_array;
	if (intrinsic_.ConvertToJsonValue(value["intrinsic"]) == false) {
		return false;
	}
	return true;
}

bool PinholeCameraTrajectory::ConvertFromJsonValue(const Json::Value &value)
{
	if (value.isObject() == false) {
		PrintWarning("PinholeCameraTrajectory read JSON failed: unsupported json format.\n");
		return false;
	}
	if (value.get("class_name", "").asString() != "PinholeCameraTrajectory" ||
			value.get("version_major", 1).asInt() != 1 ||
			value.get("version_minor", 0).asInt() != 0) {
		PrintWarning("PinholeCameraTrajectory read JSON failed: unsupported json format.\n");
		return false;
	}
	if (intrinsic_.ConvertFromJsonValue(value["intrinsic"]) == false) {
		return false;
	}
	const Json::Value &trajectory_array = value["extrinsic"];
	if (trajectory_array.size() == 0) {
		PrintWarning("PinholeCameraTrajectory read JSON failed: empty trajectory.\n");
		return false;
	}
	extrinsic_.resize(trajectory_array.size());
	for (int i = 0; i < (int)trajectory_array.size(); i++) {
		const Json::Value &status_object = trajectory_array[i];
		if (EigenMatrix4dFromJsonArray(extrinsic_[i], status_object) == false) {
			return false;
		}
	}
	return true;
}

}	// namespace three
