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
#include <jsoncpp/include/json/json.h>
#include <Core/Utility/Console.h>

namespace three{

namespace {

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
	return trajectory.ConvertFromJsonValue(root_object);
}

bool WriteViewTrajectoryToJSONStream(std::ostream &json_stream,
		const ViewTrajectory &trajectory)
{
	Json::Value root_object;
	if (trajectory.ConvertToJsonValue(root_object) == false) {
		return false;
	}
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
