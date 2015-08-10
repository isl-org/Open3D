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

#include "PointCloudIO.h"

#include <unordered_map>

namespace three{

namespace {

const std::unordered_map<std::string,
		std::function<bool(const std::string &, PointCloud &)>>
		file_extension_to_pointcloud_read_function
		{{".xyz", ReadPointCloudFromXYZ},
		{".xyzn", ReadPointCloudFromXYZN},
		{".ply", ReadPointCloudFromPLY},
		{".pcd", ReadPointCloudFromPCD},
		};

const std::unordered_map<std::string,
		std::function<bool(const std::string &, const PointCloud &,
		const bool, const bool)>>
		file_extension_to_pointcloud_write_function
		{{".xyz", WritePointCloudToXYZ},
		{".xyzn", WritePointCloudToXYZN},
		{".ply", WritePointCloudToPLY},
		{".pcd", WritePointCloudToPCD},
		};
}	// unnamed namespace

bool ReadPointCloud(const std::string &filename, PointCloud &pointcloud)
{
	size_t dot_pos = filename.find_last_of(".");
	if (dot_pos == std::string::npos) {
		PrintDebug("Read PointCloud failed: unknown file extension.");
		return false;
	}
	std::string filename_ext = filename.substr(dot_pos);
	auto map_itr =
			file_extension_to_pointcloud_read_function.find(filename_ext);
	if (map_itr == file_extension_to_pointcloud_read_function.end()) {
		PrintDebug("Read PointCloud failed: unknown file extension.");
		return false;
	}
	return map_itr->second(filename, pointcloud);
}

bool WritePointCloud(const std::string &filename, const PointCloud &pointcloud,
		const bool write_ascii/* = false*/, const bool compressed/* = false*/)
{
	size_t dot_pos = filename.find_last_of(".");
	if (dot_pos == std::string::npos) {
		PrintDebug("Write PointCloud failed: unknown file extension.");
		return false;
	}
	std::string filename_ext = filename.substr(dot_pos);
	auto map_itr =
	file_extension_to_pointcloud_write_function.find(filename_ext);
	if (map_itr == file_extension_to_pointcloud_write_function.end()) {
		PrintDebug("Write PointCloud failed: unknown file extension.");
		return false;
	}
	return map_itr->second(filename, pointcloud, write_ascii, compressed);
}

}	// namespace three
