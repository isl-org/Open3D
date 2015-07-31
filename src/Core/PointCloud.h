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
	bool HasPoints() const {
		return points_.size() > 0;
	}

	bool HasNormals() const {
		return points_.size() > 0 && normals_.size() == points_.size();
	}

	bool HasColors() const {
		return points_.size() > 0 && colors_.size() == points_.size();
	}

	void Clear() { points_.clear(); normals_.clear(); colors_.clear(); }
	
	void CloneFrom(const PointCloud &reference);

public:
	std::vector<Eigen::Vector3d> points_;
	std::vector<Eigen::Vector3d> normals_;
	std::vector<Eigen::Vector3d> colors_;
};

}	// namespace three
