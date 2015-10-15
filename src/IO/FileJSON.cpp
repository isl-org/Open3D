// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
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

#include "ViewTrajectoryIO.h"

#include <fstream>
#include <External/jsoncpp/include/json/json.h>

namespace three{

namespace {

Json::Value EigenVector3DToJsonArray(const Eigen::Vector3d &v)
{
	Json::Value array;
	array.append(v(0));
	array.append(v(1));
	array.append(v(2));
	return array;
}

}	// unnamed namespace

bool ReadViewTrajectoryFromJSON(const std::string &filename,
		ViewTrajectory &trajectory)
{
	return true;
}

bool WriteViewTrajectoryToJSON(const std::string &filename,
		const ViewTrajectory &trajectory)
{
	std::ofstream file_out(filename);
	if (file_out.is_open() == false) {
		PrintWarning("Write JSON failed: unable to open file.\n");
		return false;
	}
	Json::Value trajectory_array;
	for (size_t i = 0; i < trajectory.view_status_.size(); i++) {
		const auto &status = trajectory.view_status_[i];
		Json::Value status_object;
		status_object["field_of_view"] = status.field_of_view;
		status_object["zoom"] = status.zoom;
		status_object["lookat"] = EigenVector3DToJsonArray(status.lookat);
		status_object["up"] = EigenVector3DToJsonArray(status.up);
		status_object["front"] = EigenVector3DToJsonArray(status.front);
		trajectory_array.append(status_object);
	}

	Json::Value root_object;
	root_object["class_name"] = "ViewTrajectory";
	root_object["version_major"] = 1;
	root_object["version_minor"] = 0;
	root_object["is_loop"] = trajectory.is_loop_;
	root_object["interval"] = trajectory.interval_;
	root_object["trajectory"] = trajectory_array;

	Json::StyledStreamWriter writer;
	writer.write(file_out, root_object);
	file_out.close();
	return true;
}

}	// namespace three
