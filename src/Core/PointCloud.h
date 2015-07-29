#pragma once

#include <vector>
#include <Eigen/Core>

namespace three {

class PointCloud
{
public:
	PointCloud(void);
	~PointCloud(void);
	
public:
	bool HasNormal() {
		return points_.size() > 0 && normals_.size() == points_.size();
	}

	bool HasColor() {
		return points_.size() > 0 && colors_.size() == points_.size();
	}

public:
	std::vector<Eigen::Vector3d> points_;
	std::vector<Eigen::Vector3d> normals_;
	std::vector<Eigen::Vector3d> colors_;
};

}	// namespace three
