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

#include "ViewTrajectoryIO.h"

#include <fstream>
#include <sstream>
#include <External/jsoncpp/include/json/json.h>
#include <Core/Utility/Console.h>

namespace three{

namespace {

Json::Value EigenVector3dToJsonArray(const Eigen::Vector3d &v)
{
	Json::Value array;
	array.append(v(0));
	array.append(v(1));
	array.append(v(2));
	return array;
}

Eigen::Vector3d JsonArrayToEigenVector3d(const Json::Value &v)
{
	if (v.size() != 3) {
		return Eigen::Vector3d::Zero();
	} else {
		return Eigen::Vector3d(v[0].asDouble(), v[1].asDouble(), 
				v[2].asDouble());
	}
}

bool ReadViewTrajectoryFromJSONStream(std::istream &json_stream,
		ViewTrajectory &trajectory)
{
	Json::Value root_object;
	Json::Reader reader;
	bool is_parse_successful = reader.parse(json_stream, root_object);
	if (is_parse_successful == false) {
		PrintWarning("Read JSON failed: %s.\n", 
				reader.getFormattedErrorMessages().c_str());
		return false;
	}
	if (root_object.get("class_name", "").asString() != "ViewTrajectory" ||
			root_object.get("version_major", 1).asInt() != 1 ||
			root_object.get("version_minor", 0).asInt() != 0) {
		PrintWarning("Read JSON failed: unsupported json format.\n");
		return false;
	}
	trajectory.is_loop_ = root_object.get("is_loop", false).asBool();
	trajectory.interval_ = root_object.get("interval", 29).asInt();
	const Json::Value &trajectory_array = root_object["trajectory"];
	if (trajectory_array.size() == 0) {
		PrintWarning("Read JSON failed: empty trajectory.\n");
		return false;
	}
	trajectory.view_status_.resize(trajectory_array.size());
	for (int i = 0; i < (int)trajectory_array.size(); i++) {
		const Json::Value &status_object = trajectory_array[i];
		ViewTrajectory::ViewStatus status;
		status.field_of_view = 
				status_object.get("field_of_view", 60.0).asDouble();
		status.zoom = status_object.get("zoom", 0.7).asDouble();
		status.lookat = JsonArrayToEigenVector3d(status_object["lookat"]);
		status.up = JsonArrayToEigenVector3d(status_object["up"]);
		status.front = JsonArrayToEigenVector3d(status_object["front"]);
		trajectory.view_status_[i] = status;
	}
	return true;
}

bool WriteViewTrajectoryToJSONStream(std::ostream &json_stream,
		const ViewTrajectory &trajectory)
{
	Json::Value trajectory_array;
	for (const auto &status : trajectory.view_status_) {
		Json::Value status_object;
		status_object["field_of_view"] = status.field_of_view;
		status_object["zoom"] = status.zoom;
		status_object["lookat"] = EigenVector3dToJsonArray(status.lookat);
		status_object["up"] = EigenVector3dToJsonArray(status.up);
		status_object["front"] = EigenVector3dToJsonArray(status.front);
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
	writer.write(json_stream, root_object);
	return true;
}

}	// unnamed namespace

bool ReadViewTrajectoryFromJSON(const std::string &filename,
		ViewTrajectory &trajectory)
{
	std::ifstream file_in(filename);
	if (file_in.is_open() == false) {
		PrintWarning("Read JSON failed: unable to open file.\n");
		return false;
	}
	bool success = ReadViewTrajectoryFromJSONStream(file_in, trajectory);
	file_in.close();
	return success;
}

bool WriteViewTrajectoryToJSON(const std::string &filename,
		const ViewTrajectory &trajectory)
{
	std::ofstream file_out(filename);
	if (file_out.is_open() == false) {
		PrintWarning("Write JSON failed: unable to open file.\n");
		return false;
	}
	bool success = WriteViewTrajectoryToJSONStream(file_out, trajectory);
	file_out.close();
	return success;
}

bool ReadViewTrajectoryFromJSONString(const std::string &json_string,
		ViewTrajectory &trajectory)
{
	std::istringstream iss(json_string);
	return ReadViewTrajectoryFromJSONStream(iss, trajectory);
}

bool WriteViewTrajectoryToJSONString(std::string &json_string,
		const ViewTrajectory &trajectory)
{
	std::ostringstream oss;
	bool success = WriteViewTrajectoryToJSONStream(oss, trajectory);
	json_string = oss.str();
	return success;
}

}	// namespace three
