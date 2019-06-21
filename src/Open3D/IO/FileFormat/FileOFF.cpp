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

#include <fstream>
#include <vector>

#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace io {

bool ReadTriangleMeshFromOFF(const std::string &filename,
                             geometry::TriangleMesh &mesh) {
    FILE *file_in;

    if ((file_in = fopen(filename.c_str(), "r")) == NULL) {
        utility::PrintWarning("Read OFF failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }

    char header[3];
    int num_of_vertices, num_of_triangles, num_of_edges;
    fscanf(file_in, "%s", header);
    fscanf(file_in, "%d %d %d", &num_of_vertices, &num_of_triangles,
           &num_of_edges);
    if (num_of_vertices == 0 || num_of_triangles == 0) {
        utility::PrintWarning("Read OFF failed: unable to read header.\n");
        return false;
    }

    mesh.Clear();
    mesh.vertices_.resize(num_of_vertices);
    mesh.triangles_.resize(num_of_triangles);

    utility::ResetConsoleProgress(num_of_vertices + num_of_triangles,
                                  "Reading OFF: ");

    float vx, vy, vz;
    for (int vidx = 0; vidx < num_of_vertices; vidx++) {
        fscanf(file_in, "%f %f %f", &vx, &vy, &vz);
        mesh.vertices_[vidx] = Eigen::Vector3d(vx, vy, vz);
        utility::AdvanceConsoleProgress();
    }

    int n, vidx1, vidx2, vidx3;
    for (int tidx = 0; tidx < num_of_triangles; tidx++) {
        fscanf(file_in, "%d %d %d %d", &n, &vidx1, &vidx2, &vidx3);
        if (n != 3) {
            utility::PrintWarning("Read OFF failed: not a triangle mesh.\n");
            return false;
        }
        mesh.triangles_[tidx] = Eigen::Vector3i(vidx1, vidx2, vidx3);
        utility::AdvanceConsoleProgress();
    }

    fclose(file_in);
    return true;
}

bool WriteTriangleMeshToOFF(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/) {
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);

    if (!file) {
        utility::PrintWarning("Write OFF failed: unable to open file.\n");
        return false;
    }

    size_t num_of_vertices = mesh.vertices_.size();
    size_t num_of_triangles = mesh.triangles_.size();
    if (num_of_vertices == 0 || num_of_triangles == 0) {
        utility::PrintWarning("Write OFF failed: empty file.\n");
        return false;
    }

    file << "OFF\n";
    file << num_of_vertices << " " << num_of_triangles << " 0\n";

    utility::ResetConsoleProgress(num_of_vertices + num_of_triangles,
                                  "Writing OFF: ");
    for (int vidx = 0; vidx < num_of_vertices; ++vidx) {
        const Eigen::Vector3d &vertex = mesh.vertices_[vidx];
        file << vertex(0) << " " << vertex(1) << " " << vertex(2) << "\n";
        utility::AdvanceConsoleProgress();
    }

    for (int tidx = 0; tidx < num_of_triangles; ++tidx) {
        const Eigen::Vector3i &triangle = mesh.triangles_[tidx];
        file << "3 " << triangle(0) << " " << triangle(1) << " " << triangle(2)
             << "\n";
        utility::AdvanceConsoleProgress();
    }

    return true;
}

}  // namespace io
}  // namespace open3d