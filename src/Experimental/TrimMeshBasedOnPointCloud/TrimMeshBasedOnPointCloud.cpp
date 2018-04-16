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

#include <Core/Core.h>
#include <IO/IO.h>

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > TrimMeshBasedOnPointCloud [options]\n");
	printf("      Trim a mesh baesd on distance to a point cloud.\n");
	printf("\n");
	printf("Basic options:\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --verbose n               : Set verbose level (0-4). Default: 2.\n");
	printf("    --in_mesh mesh_file       : Input mesh file. MUST HAVE.\n");
	printf("    --out_mesh mesh_file      : Output mesh file. MUST HAVE.\n");
	printf("    --pointcloud pcd_file     : Reference pointcloud file. MUST HAVE.\n");
	printf("    --distance d              : Maximum distance. MUST HAVE.\n");
}

int main(int argc, char *argv[])
{
	using namespace three;

	if (argc < 4 || ProgramOptionExists(argc, argv, "--help") ||
			ProgramOptionExists(argc, argv, "-h")) {
		PrintHelp();
		return 0;
	}
	int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
	SetVerbosityLevel((VerbosityLevel)verbose);
	auto in_mesh_file = GetProgramOptionAsString(argc, argv, "--in_mesh");
	auto out_mesh_file = GetProgramOptionAsString(argc, argv, "--out_mesh");
	auto pcd_file = GetProgramOptionAsString(argc, argv, "--pointcloud");
	auto distance = GetProgramOptionAsDouble(argc, argv, "--distance");
	if (distance <= 0.0) {
		PrintWarning("Illegal distance.\n");
		return 0;
	}
	if (in_mesh_file.empty() || out_mesh_file.empty() || pcd_file.empty()) {
		PrintWarning("Missing file names.\n");
		return 0;
	}
	auto mesh = CreateMeshFromFile(in_mesh_file);
	auto pcd = CreatePointCloudFromFile(pcd_file);
	if (mesh->IsEmpty() || pcd->IsEmpty()) {
		PrintWarning("Empty geometry.\n");
		return 0;
	}

	KDTreeFlann kdtree;
	kdtree.SetGeometry(*pcd);
	std::vector<bool> remove_vertex_mask(mesh->vertices_.size(), false);
	ResetConsoleProgress(mesh->vertices_.size(), "Prune vetices: ");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)mesh->vertices_.size(); i++) {
		std::vector<int> indices(1);
		std::vector<double> dists(1);
		int k = kdtree.SearchKNN(mesh->vertices_[i], 1, indices, dists);
		if (k == 0 || dists[0] > distance * distance) {
			remove_vertex_mask[i] = true;
		}
#ifdef _OPENMP
#pragma omp critical
#endif
		{
			AdvanceConsoleProgress();
		}
	}

	std::vector<int> index_old_to_new(mesh->vertices_.size());
	bool has_vert_normal = mesh->HasVertexNormals();
	bool has_vert_color = mesh->HasVertexColors();
	size_t old_vertex_num = mesh->vertices_.size();
	size_t k = 0;											// new index
	bool has_tri_normal = mesh->HasTriangleNormals();
	size_t old_triangle_num = mesh->triangles_.size();
	size_t kt = 0;
	for (size_t i = 0; i < old_vertex_num; i++) {			// old index
		if (remove_vertex_mask[i] == false) {
			mesh->vertices_[k] = mesh->vertices_[i];
			if (has_vert_normal)
				mesh->vertex_normals_[k] = mesh->vertex_normals_[i];
			if (has_vert_color)
				mesh->vertex_colors_[k] = mesh->vertex_colors_[i];
			index_old_to_new[i] = (int)k;
			k++;
		} else {
			index_old_to_new[i] = -1;
		}
	}
	mesh->vertices_.resize(k);
	if (has_vert_normal) mesh->vertex_normals_.resize(k);
	if (has_vert_color) mesh->vertex_colors_.resize(k);
	if (k < old_vertex_num) {
		for (size_t i = 0; i < old_triangle_num; i++) {
			auto &triangle = mesh->triangles_[i];
			triangle(0) = index_old_to_new[triangle(0)];
			triangle(1) = index_old_to_new[triangle(1)];
			triangle(2) = index_old_to_new[triangle(2)];
			if (triangle(0) != -1 && triangle(1) != -1 && triangle(2) != -1) {
				mesh->triangles_[kt] = mesh->triangles_[i];
				if (has_tri_normal)
					mesh->triangle_normals_[kt] = mesh->triangle_normals_[i];
				kt++;
			}
		}
		mesh->triangles_.resize(kt);
		if (has_tri_normal) mesh->triangle_normals_.resize(kt);
	}
	PrintDebug("[TrimMeshBasedOnPointCloud] %d vertices and %d triangles have been removed.\n",
			old_vertex_num - k, old_triangle_num - kt);
	WriteTriangleMesh(out_mesh_file, *mesh);
	return 1;
}
