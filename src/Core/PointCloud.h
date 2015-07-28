#pragma once

#include <vector>
#include <Eigen/Core>

class PointCloud
{
public:
	PointCloud(void);
	~PointCloud(void);
	
public:
	bool HasNormal() {
		return normals_.size() == 0;
	}

public:
	std::vector<Eigen::Vector3d> points_;
	std::vector<Eigen::Vector3d> normals_;
};
