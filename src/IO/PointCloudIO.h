#pragma once

#include <string>
#include <Core/PointCloud.h>

namespace three {

/// The general entrance for reading a PointCloud from a file
/// The function calls read functions based on the extension name of filename.
/// \return If the read function is successful. 
bool ReadPointCloud(const std::string &filename, PointCloud &pointcloud);

/// The general entrance for writing a PointCloud to a file
/// The function calls write functions based on the extension name of filename.
/// \return If the write function is successful. 
bool WritePointCloud(const std::string &filename, const PointCloud &pointcloud);

bool ReadPointCloudFromXYZ(
		const std::string &filename,
		PointCloud &pointcloud);

bool WritePointCloudToXYZ(
		const std::string &filename,
		const PointCloud &pointcloud);

bool ReadPointCloudFromXYZN(
		const std::string &filename,
		PointCloud &pointcloud);

bool WritePointCloudToXYZN(
		const std::string &filename,
		const PointCloud &pointcloud);

bool ReadPointCloudFromPLY(
		const std::string &filename,
		PointCloud &pointcloud);

bool WritePointCloudToPLY(
		const std::string &filename,
		const PointCloud &pointcloud,
		const bool write_ascii = false);

}	// namespace three
