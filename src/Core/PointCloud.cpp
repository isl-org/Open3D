#include "PointCloud.h"

namespace three{

PointCloud::PointCloud(void)
{
}

PointCloud::~PointCloud(void)
{
}
	
void PointCloud::CloneFrom(const PointCloud &reference)
{
	Clear();

	points_.resize(reference.points_.size());
	for (size_t i = 0; i < reference.points_.size(); i++) {
		points_[i] = reference.points_[i];
	}

	normals_.resize(reference.normals_.size());
	for (size_t i = 0; i < reference.normals_.size(); i++) {
		normals_[i] = reference.normals_[i];
	}

	colors_.resize(reference.colors_.size());
	for (size_t i = 0; i < reference.colors_.size(); i++) {
		colors_[i] = reference.colors_[i];
	}
}

}	// namespace three
