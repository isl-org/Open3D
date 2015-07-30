#include "Config.h"
#include "PointCloudIO.h"

namespace three{

bool ReadPointCloudFromXYZ(
		const std::string &filename,
		PointCloud &pointcloud)
{
	FILE *file = fopen(filename.c_str(), "r");
	if (file != NULL) {
		char line_buffer[DEFAULT_IO_BUFFER_SIZE];
		double x, y, z;

		while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
			if (sscanf(line_buffer, "%ld %ld %ld", &x, &y, &z) == 3) {
				pointcloud.points_.push_back(Eigen::Vector3d(x, y, z));
			}
		}

		fclose(file);
		return true;
	} else { // if (file != NULL)
		return false;
	}
}

bool WritePointCloudToXYZ(
		const std::string &filename,
		const PointCloud &pointcloud)
{
	FILE *file = fopen(filename.c_str(), "w");
	if (file != NULL) {
		for (size_t i = 0; i < pointcloud.points_.size(); i++) {
			const Eigen::Vector3d &point = pointcloud.points_[i];
			if (fprintf(file, "%.10f %.10f %.10f\n", point(0), point(1), point(2)) < 0) {
				return false;	// error happens during writting.
			}
		}

		fclose(file);
		return true;
	} else {	// if (file != NULL)
		return false;
	}
}

}	// namespace three
