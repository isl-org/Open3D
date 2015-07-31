#include "Config.h"
#include "PointCloudIO.h"

#include <cstdio>

namespace three{

bool ReadPointCloudFromXYZN(
		const std::string &filename,
		PointCloud &pointcloud)
{
	FILE *file = fopen(filename.c_str(), "r");
	if (file == NULL) {
		return false;
	}

	char line_buffer[DEFAULT_IO_BUFFER_SIZE];
	double x, y, z, nx, ny, nz;
	pointcloud.Clear();

	while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
		if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf",
				&x, &y, &z, &nx, &ny, &nz) == 6) 
		{
			pointcloud.points_.push_back(Eigen::Vector3d(x, y, z));
			pointcloud.normals_.push_back(Eigen::Vector3d(nx, ny, nz));
		}
	}

	fclose(file);
	return true;
}

bool WritePointCloudToXYZN(
		const std::string &filename,
		const PointCloud &pointcloud)
{
	if (pointcloud.HasNormal() == false) {
		return false;
	}

	FILE *file = fopen(filename.c_str(), "w");
	if (file == NULL) {
		return false;
	}

	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		const Eigen::Vector3d &point = pointcloud.points_[i];
		const Eigen::Vector3d &normal = pointcloud.normals_[i];
		if (fprintf(file, "%.10f %.10f %.10f %.10f %.10f %.10f\n",
				point(0), point(1), point(2), 
				normal(0), normal(1), normal(2)) < 0)
		{
			return false;	// error happens during writting.
		}
	}

	fclose(file);
	return true;
}

}	// namespace three
