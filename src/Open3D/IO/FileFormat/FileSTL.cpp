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

bool ReadTriangleMeshFromSTL(const std::string &filename,
                             geometry::TriangleMesh &mesh,
                             bool print_progress) {
    FILE *myFile = fopen(filename.c_str(), "rb");

    if (!myFile) {
        utility::LogWarning("Read STL failed: unable to open file.\n");
        fclose(myFile);
        return false;
    }

    int num_of_triangles;
    if (myFile) {
        char header[80] = "";
        fread(header, sizeof(char), 80, myFile);
        fread(&num_of_triangles, sizeof(unsigned int), 1, myFile);
    } else {
        utility::LogWarning("Read STL failed: unable to read header.\n");
        fclose(myFile);
        return false;
    }

    if (num_of_triangles == 0) {
        utility::LogWarning("Read STL failed: empty file.\n");
        fclose(myFile);
        return false;
    }

    mesh.vertices_.clear();
    mesh.triangles_.clear();
    mesh.triangle_normals_.clear();
    mesh.vertices_.resize(num_of_triangles * 3);
    mesh.triangles_.resize(num_of_triangles);
    mesh.triangle_normals_.resize(num_of_triangles);

    utility::ConsoleProgressBar progress_bar(num_of_triangles,
                                             "Reading STL: ", print_progress);
    for (int i = 0; i < num_of_triangles; i++) {
        char buffer[50];
        float *float_buffer;
        if (myFile) {
            fread(buffer, sizeof(char), 50, myFile);
            float_buffer = reinterpret_cast<float *>(buffer);
            mesh.triangle_normals_[i] =
                    Eigen::Map<Eigen::Vector3f>(float_buffer).cast<double>();
            for (int j = 0; j < 3; j++) {
                float_buffer = reinterpret_cast<float *>(buffer + 12 * (j + 1));
                mesh.vertices_[i * 3 + j] =
                        Eigen::Map<Eigen::Vector3f>(float_buffer)
                                .cast<double>();
            }
            mesh.triangles_[i] =
                    Eigen::Vector3i(i * 3 + 0, i * 3 + 1, i * 3 + 2);
            // ignore buffer[48] and buffer [49] because it is rarely used.

        } else {
            utility::LogWarning("Read STL failed: not enough triangles.\n");
            fclose(myFile);
            return false;
        }
        ++progress_bar;
    }

    fclose(myFile);
    return true;
}

bool WriteTriangleMeshToSTL(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/,
                            bool write_vertex_normals /* = true*/,
                            bool write_vertex_colors /* = true*/,
                            bool print_progress) {
    std::ofstream myFile(filename.c_str(), std::ios::out | std::ios::binary);

    if (!myFile) {
        utility::LogWarning("Write STL failed: unable to open file.\n");
        return false;
    }

    if (!mesh.HasTriangleNormals()) {
        utility::LogWarning("Write STL failed: compute normals first.\n");
        return false;
    }

    size_t num_of_triangles = mesh.triangles_.size();
    if (num_of_triangles == 0) {
        utility::LogWarning("Write STL failed: empty file.\n");
        return false;
    }
    char header[80] = "Created by Open3D";
    myFile.write(header, 80);
    myFile.write((char *)(&num_of_triangles), 4);

    utility::ConsoleProgressBar progress_bar(num_of_triangles,
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
