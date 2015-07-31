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

#include "Config.h"
#include "PointCloudIO.h"

#include <cstdio>

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

	while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
		if (line_buffer[0] == 0 || line_buffer[0] == '#') {
			continue;
		}
		const std::string line_str(line_buffer);
		if (size_t pos = line_str.find_first_of(" \t\r") != std::string::npos) {
			std::string line_type = line_str.substr(0, pos);
			if (line_type == "VERSION") {
				continue;
			}
			if (line_type == "FIELDS" || line_type == "COLUMNS") {
				continue;
			}
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
