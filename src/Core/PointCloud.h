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
		return normals_.size() == points_.size();
	}

public:
	std::vector<Eigen::Vector3d> points_;
	std::vector<Eigen::Vector3d> normals_;
};

}	// namespace three
