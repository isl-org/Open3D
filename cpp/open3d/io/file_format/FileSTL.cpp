// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <fstream>
#include <vector>

#include "open3d/io/FileFormatIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressBar.h"

namespace open3d {
namespace io {

FileGeometry ReadFileGeometryTypeSTL(const std::string &path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

bool WriteTriangleMeshToSTL(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/,
                            bool write_vertex_normals /* = true*/,
                            bool write_vertex_colors /* = true*/,
                            bool write_triangle_uvs /* = true*/,
                            bool print_progress) {
    if (write_triangle_uvs && mesh.HasTriangleUvs()) {
        utility::LogWarning(
                "This file format does not support writing textures and uv "
                "coordinates. Consider using .obj");
    }
    if (write_ascii) {
        utility::LogError("Writing ascii STL file is not supported yet.");
    }

    std::ofstream myFile(filename.c_str(), std::ios::out | std::ios::binary);

    if (!myFile) {
        utility::LogWarning("Write STL failed: unable to open file.");
        return false;
    }

    if (!mesh.HasTriangleNormals()) {
        utility::LogWarning("Write STL failed: compute normals first.");
        return false;
    }

    size_t num_of_triangles = mesh.triangles_.size();
    if (num_of_triangles == 0) {
        utility::LogWarning("Write STL failed: empty file.");
        return false;
    }
    char header[80] = "Created by Open3D";
    myFile.write(header, 80);
    myFile.write((char *)(&num_of_triangles), 4);

    utility::ProgressBar progress_bar(num_of_triangles,
                                      "Writing STL: ", print_progress);
    for (size_t i = 0; i < num_of_triangles; i++) {
        Eigen::Vector3f float_vector3f =
                mesh.triangle_normals_[i].cast<float>();
        myFile.write(reinterpret_cast<const char *>(float_vector3f.data()), 12);
        for (int j = 0; j < 3; j++) {
            Eigen::Vector3f float_vector3f =
                    mesh.vertices_[mesh.triangles_[i][j]].cast<float>();
            myFile.write(reinterpret_cast<const char *>(float_vector3f.data()),
                         12);
        }
        char blank[2] = {0, 0};
        myFile.write(blank, 2);
        ++progress_bar;
    }
    return true;
}

}  // namespace io
}  // namespace open3d
