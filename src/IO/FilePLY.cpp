#include "Config.h"
#include "PointCloudIO.h"

#include <Core/Console.h>
#include <External/rply/rply.h>

namespace three{

namespace {

struct PLYReaderState {
	PointCloud *pointcloud_ptr;
	long vertex_index;
	long vertex_num;
	long normal_index;
	long normal_num;
	long color_index;
	long color_num;
};

int ReadVertexCallback(p_ply_argument argument)
{
	PLYReaderState *state_ptr;
	long index;
	ply_get_argument_user_data(argument, 
			reinterpret_cast<void **>(&state_ptr), &index);
	double value = ply_get_argument_value(argument);
	state_ptr->pointcloud_ptr->points_[state_ptr->vertex_index](index) = value;
	if (index == 2) {	// reading 'z'
		state_ptr->vertex_index++;
		if (state_ptr->vertex_index > state_ptr->vertex_num) {
			return 0;
		}
	}
	return 1;
}

int ReadNormalCallback(p_ply_argument argument)
{
	PLYReaderState *state_ptr;
	long index;
	ply_get_argument_user_data(argument, 
			reinterpret_cast<void **>(&state_ptr), &index);
	double value = ply_get_argument_value(argument);
	state_ptr->pointcloud_ptr->points_[state_ptr->normal_index](index) = value;
	if (index == 2) {	// reading 'z'
		state_ptr->normal_index++;
		if (state_ptr->normal_index > state_ptr->normal_num) {
			return 0;
		}
	}
	return 1;
}

int ReadColorCallback(p_ply_argument argument)
{
	PLYReaderState *state_ptr;
	long index;
	ply_get_argument_user_data(argument, 
			reinterpret_cast<void **>(&state_ptr), &index);
	double value = ply_get_argument_value(argument);
	state_ptr->pointcloud_ptr->points_[state_ptr->color_index](index) = value / 255.0;
	if (index == 2) {	// reading 'z'
		state_ptr->color_index++;
		if (state_ptr->color_index > state_ptr->color_num) {
			return 0;
		}
	}
	return 1;
}

int WriteVertexCallback(p_ply_argument argument) {
	return 1;
}

}	// unnamed namespace

bool ReadPointCloudFromPLY(
		const std::string &filename,
		PointCloud &pointcloud)
{
	p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
	if (!ply_file) {
		PrintDebug("Open PLY failed: unable to open file.\n");
		return false;
	}
	if (!ply_read_header(ply_file)) {
		PrintDebug("Open PLY failed: unable to parse header.\n");
		return false;
	}

	PLYReaderState state;
	state.pointcloud_ptr = &pointcloud;
	state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x", 
			ReadVertexCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "y",  ReadVertexCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "z",  ReadVertexCallback, &state, 2);

	state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx", 
			ReadNormalCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "ny",  ReadNormalCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "nz",  ReadNormalCallback, &state, 2);

	state.color_num = ply_set_read_cb(ply_file, "vertex", "red", 
			ReadColorCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "green",  ReadColorCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "blue",  ReadColorCallback, &state, 2);

	if (state.vertex_num <= 0) {
		PrintDebug("Open PLY failed: number of vertex <= 0.\n");
		return false;
	}

	state.vertex_index = 0;
	state.normal_index = 0;
	state.color_index = 0;

	pointcloud.Clear();
	pointcloud.points_.resize(state.vertex_num);
	pointcloud.normals_.resize(state.normal_num);
	pointcloud.colors_.resize(state.color_num);

	if (!ply_read(ply_file)) {
		PrintDebug("Open PLY failed: unable to read file.\n");
		return false;
	}

	ply_close(ply_file);
	return true;
}

bool WritePointCloudToPLY(
		const std::string &filename,
		const PointCloud &pointcloud)
{
	return true;
}

}	// namespace three
