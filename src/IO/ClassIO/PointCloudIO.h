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

#pragma once

#include <string>
#include <Core/Geometry/PointCloud.h>

namespace three {

/// The general entrance for reading a PointCloud from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadPointCloud(const std::string &filename, PointCloud &pointcloud);

/// The general entrance for writing a PointCloud to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// \return return true if the write function is successful, false otherwise.
bool WritePointCloud(const std::string &filename, const PointCloud &pointcloud,
		bool write_ascii = false, bool compressed = false);

bool ReadPointCloudFromXYZ(const std::string &filename, PointCloud &pointcloud);

bool WritePointCloudToXYZ(const std::string &filename,
		const PointCloud &pointcloud, bool write_ascii = false,
		bool compressed = false);

bool ReadPointCloudFromXYZN(const std::string &filename,
		PointCloud &pointcloud);

bool WritePointCloudToXYZN(const std::string &filename,
		const PointCloud &pointcloud, bool write_ascii = false,
		bool compressed = false);

bool ReadPointCloudFromXYZRGB(const std::string &filename,
		PointCloud &pointcloud);

bool WritePointCloudToXYZRGB(const std::string &filename,
		const PointCloud &pointcloud, bool write_ascii = false,
		bool compressed = false);

bool ReadPointCloudFromPLY(
		const std::string &filename,
		PointCloud &pointcloud);

bool WritePointCloudToPLY(const std::string &filename,
		const PointCloud &pointcloud, bool write_ascii = false,
		bool compressed = false);

bool ReadPointCloudFromPCD(const std::string &filename, PointCloud &pointcloud);

bool WritePointCloudToPCD(const std::string &filename,
		const PointCloud &pointcloud, bool write_ascii = false,
		bool compressed = false);

bool ReadPointCloudFromPTS(const std::string &filename, PointCloud &pointcloud);

bool WritePointCloudToPTS(const std::string &filename,
		const PointCloud &pointcloud, bool write_ascii = false,
		bool compressed = false);

}	// namespace three
