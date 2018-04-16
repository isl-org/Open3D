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

#include "PointCloudIO.h"

#include <unordered_map>
#include <Core/Utility/Console.h>
#include <Core/Utility/FileSystem.h>

namespace three{

namespace {

static const std::unordered_map<std::string,
		std::function<bool(const std::string &, PointCloud &)>>
		file_extension_to_pointcloud_read_function
		{{"xyz", ReadPointCloudFromXYZ},
		{"xyzn", ReadPointCloudFromXYZN},
		{"xyzrgb", ReadPointCloudFromXYZRGB},
		{"ply", ReadPointCloudFromPLY},
		{"pcd", ReadPointCloudFromPCD},
		{"pts", ReadPointCloudFromPTS},
		};

static const std::unordered_map<std::string,
		std::function<bool(const std::string &, const PointCloud &,
		const bool, const bool)>>
		file_extension_to_pointcloud_write_function
		{{"xyz", WritePointCloudToXYZ},
		{"xyzn", WritePointCloudToXYZN},
		{"xyzrgb", WritePointCloudToXYZRGB},
		{"ply", WritePointCloudToPLY},
		{"pcd", WritePointCloudToPCD},
		{"pts", WritePointCloudToPTS},
		};
}	// unnamed namespace

bool ReadPointCloud(const std::string &filename, PointCloud &pointcloud)
{
	std::string filename_ext =
			filesystem::GetFileExtensionInLowerCase(filename);
	if (filename_ext.empty()) {
		PrintWarning("Read PointCloud failed: unknown file extension.\n");
		return false;
	}
	auto map_itr =
			file_extension_to_pointcloud_read_function.find(filename_ext);
	if (map_itr == file_extension_to_pointcloud_read_function.end()) {
		PrintWarning("Read PointCloud failed: unknown file extension.\n");
		return false;
	}
	bool success = map_itr->second(filename, pointcloud);
	PrintDebug("Read PointCloud: %d vertices.\n",
			(int)pointcloud.points_.size());
	return success;
}

bool WritePointCloud(const std::string &filename, const PointCloud &pointcloud,
		bool write_ascii/* = false*/, bool compressed/* = false*/)
{
	std::string filename_ext =
			filesystem::GetFileExtensionInLowerCase(filename);
	if (filename_ext.empty()) {
		PrintWarning("Write PointCloud failed: unknown file extension.\n");
		return false;
	}
	auto map_itr =
			file_extension_to_pointcloud_write_function.find(filename_ext);
	if (map_itr == file_extension_to_pointcloud_write_function.end()) {
		PrintWarning("Write PointCloud failed: unknown file extension.\n");
		return false;
	}
	bool success = map_itr->second(filename, pointcloud, write_ascii,
			compressed);
	PrintDebug("Write PointCloud: %d vertices.\n",
			(int)pointcloud.points_.size());
	return success;
}

}	// namespace three
