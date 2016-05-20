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

#include "PinholeCameraTrajectory.h"

#include <jsoncpp/include/json/json.h>
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
	for (const auto &status : camera_poses_) {
		Json::Value status_object;
		if (status.ConvertToJsonValue(status_object) == false) {
			return false;
		}
		trajectory_array.append(status_object);
	}
	value["class_name"] = "PinholeCameraTrajectory";
	value["version_major"] = 1;
	value["version_minor"] = 0;
	value["constant_intrinsic"] = constant_intrinsic_;
	value["trajectory"] = trajectory_array;
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
	constant_intrinsic_ = value.get("constant_intrinsic", false).asBool();
	const Json::Value &trajectory_array = value["trajectory"];
	if (trajectory_array.size() == 0) {
		PrintWarning("PinholeCameraTrajectory read JSON failed: empty trajectory.\n");
		return false;
	}
	camera_poses_.resize(trajectory_array.size());
	for (int i = 0; i < (int)trajectory_array.size(); i++) {
		const Json::Value &status_object = trajectory_array[i];
		PinholeCameraParameters status;
		if (status.ConvertFromJsonValue(status_object) == false) {
			return false;
		}
		camera_poses_[i] = status;
	}
	return true;
}

}	// namespace three
