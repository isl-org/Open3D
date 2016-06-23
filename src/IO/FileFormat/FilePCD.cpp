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

#include <IO/ClassIO/PointCloudIO.h>

#include <cstdio>
#include <sstream>
#include <Core/Utility/Console.h>
#include <Core/Utility/Helper.h>

// References for PCD file IO
// http://pointclouds.org/documentation/tutorials/pcd_file_format.php
// https://github.com/PointCloudLibrary/pcl/blob/master/io/src/pcd_io.cpp
// https://www.mathworks.com/matlabcentral/fileexchange/40382-matlab-to-point-cloud-library

namespace three{

namespace {

enum PCDDataType {
	PCD_DATA_ASCII = 0,
	PCD_DATA_BINARY = 1,
	PCD_DATA_BINARY_COMPRESSED = 2
};

bool ReadPCDHeader(FILE *file, PointCloud &pointcloud, PCDDataType &data_type)
{
	char line_buffer[DEFAULT_IO_BUFFER_SIZE];
	int specified_channel_count = 0;
	bool has_points = false, has_normals = false, has_colors = false;

	while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
		std::string line(line_buffer);
		if (line == "") {
			continue;
		}
		std::vector<std::string> st;
		SplitString(st, line, "\t\r ");
		std::stringstream sstream(line);
		sstream.imbue(std::locale::classic());
		std::string line_type;
		sstream >> line_type;
		if (line_type.substr(0, 1) == "#") {
		} else if (line_type.substr(0, 7) == "VERSION") {
		} else if (line_type.substr(0, 6) == "FIELDS" || 
				line_type.substr(0, 7) == "COLUMNS") {
			specified_channel_count = (int)st.size() - 1;
		} else if (line_type.substr(0, 7) == "VERSION") {
		} else if (line_type.substr(0, 7) == "VERSION") {
		} else if (line_type.substr(0, 7) == "VERSION") {
		} else if (line_type.substr(0, 7) == "VERSION") {
		} else if (line_type.substr(0, 7) == "VERSION") {
		} else if (line_type.substr(0, 7) == "VERSION") {
		} else if (line_type.substr(0, 7) == "VERSION") {
		}
	}

	return PCD_DATA_ASCII;
}

}	// unnamed namespace

bool ReadPointCloudFromPCD(
		const std::string &filename,
		PointCloud &pointcloud)
{
	return true;
}

bool WritePointCloudToPCD(
		const std::string &filename,
		const PointCloud &pointcloud,
		const bool write_ascii/* = false*/,
		const bool compressed/* = false*/)
{
	return true;
}

}	// namespace three
