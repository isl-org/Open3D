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

#include "Open3D/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > TrimMeshBasedOnPointCloud [options]");
    utility::LogInfo("      Trim a mesh baesd on distance to a point cloud.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4). Default: 2.");
    utility::LogInfo("    --in_mesh mesh_file       : Input mesh file. MUST HAVE.");
    utility::LogInfo("    --out_mesh mesh_file      : Output mesh file. MUST HAVE.");
    utility::LogInfo("    --pointcloud pcd_file     : Reference pointcloud file. MUST HAVE.");
    utility::LogInfo("    --distance d              : Maximum distance. MUST HAVE.");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    if (argc < 4 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 1;
    }
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    auto in_mesh_file =
            utility::GetProgramOptionAsString(argc, argv, "--in_mesh");
    auto out_mesh_file =
            utility::GetProgramOptionAsString(argc, argv, "--out_mesh");
    auto pcd_file =
            utility::GetProgramOptionAsString(argc, argv, "--pointcloud");
    auto distance = utility::GetProgramOptionAsDouble(argc, argv, "--distance");
    if (distance <= 0.0) {
        utility::LogWarning("Illegal distance.");
        return 1;
    }
    if (in_mesh_file.empty() || out_mesh_file.empty() || pcd_file.empty()) {
        utility::LogWarning("Missing file names.");
        return 1;
    }
    auto mesh = io::CreateMeshFromFile(in_mesh_file);
    auto pcd = io::CreatePointCloudFromFile(pcd_file);
    if (mesh->IsEmpty() || pcd->IsEmpty()) {
        utility::LogWarning("Empty geometry.");
        return 1;
    }

    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*pcd);
    std::vector<bool> remove_vertex_mask(mesh->vertices_.size(), false);
    utility::ConsoleProgressBar progress_bar(mesh->vertices_.size(),
                                             "Prune vetices: ");
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
        { ++progress_bar; }
    }

    std::vector<int> index_old_to_new(mesh->vertices_.size());
    bool has_vert_normal = mesh->HasVertexNormals();
    bool has_vert_color = mesh->HasVertexColors();
    size_t old_vertex_num = mesh->vertices_.size();
    size_t k = 0;  // new index
    bool has_tri_normal = mesh->HasTriangleNormals();
    size_t old_triangle_num = mesh->triangles_.size();
    size_t kt = 0;
    for (size_t i = 0; i < old_vertex_num; i++) {  // old index
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
    utility::LogInfo(
            "[TrimMeshBasedOnPointCloud] {:d} vertices and {:d} triangles have "
            "been removed.",
            old_vertex_num - k, old_triangle_num - kt);
    io::WriteTriangleMesh(out_mesh_file, *mesh);
    return 0;
}
