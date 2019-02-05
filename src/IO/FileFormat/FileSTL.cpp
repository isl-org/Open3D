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

#include <IO/ClassIO/TriangleMeshIO.h>

#include <fstream>
#include <vector>
#include <Core/Utility/Console.h>

namespace open3d {

bool ReadTriangleMeshFromSTL(const std::string &filename,
                             TriangleMesh &TriangleMesh) {
    std::ifstream myFile(filename.c_str(), std::ios::in | std::ios::binary);

    if (!myFile) {
        PrintWarning("Read STL failed: unable to open file.\n");
        return false;
    }

    int num_of_triangles;
    if (myFile) {
        char header[80] = "";
        char buffer[4];
        myFile.read(header, 80);
        myFile.read(buffer, 4);
        num_of_triangles = (int)(*((unsigned long *)buffer));
        PrintInfo("header : %s\n", header);
    } else {
        PrintWarning("Read STL failed: unable to read header.\n");
        return false;
    }

    if (num_of_triangles == 0) {
        PrintWarning("Read STL failed: empty file.\n");
        return false;
    }

    TriangleMesh.vertices_.clear();
    TriangleMesh.triangles_.clear();
    TriangleMesh.triangle_normals_.clear();
    TriangleMesh.vertices_.resize(num_of_triangles * 3);
    TriangleMesh.triangles_.resize(num_of_triangles);
    TriangleMesh.triangle_normals_.resize(num_of_triangles);

    ResetConsoleProgress(num_of_triangles, "Reading STL: ");
    for (int i = 0; i < num_of_triangles; i++) {
        char buffer[50];
        float *float_buffer;
        if (myFile) {
            myFile.read(buffer, 50);

            float_buffer = reinterpret_cast<float *>(buffer);
            TriangleMesh.triangle_normals_[i] =
                Eigen::Map<Eigen::Vector3f>(float_buffer).cast<double>();
            for (int j = 0; j < 3; j++) {
                float_buffer = reinterpret_cast<float *>(buffer + 12 * (j + 1));
                TriangleMesh.vertices_[i * 3 + j] =
                    Eigen::Map<Eigen::Vector3f>(float_buffer).cast<double>();
            }
            TriangleMesh.triangles_[i] =
                Eigen::Vector3i(i * 3 + 0, i * 3 + 1, i * 3 + 2);
            // ignore buffer[48] and buffer [49] because it is rarely used.

        } else {
            PrintWarning("Read STL failed: not enough triangles.\n");
            return false;
        }
        AdvanceConsoleProgress();
    }
    return true;
}

bool WriteTriangleMeshToSTL(const std::string &filename,
                            const TriangleMesh &TriangleMesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/) {
    std::ofstream myFile(filename.c_str(), std::ios::out | std::ios::binary);

    if (!myFile) {
        PrintWarning("Write STL failed: unable to open file.\n");
        return false;
    }

    unsigned long num_of_triangles = TriangleMesh.triangles_.size();
    if (num_of_triangles == 0) {
        PrintWarning("Write STL failed: empty file.\n");
        return false;
    }
    char header[80] = "Created by Open3D";
    myFile.write(header, 80);
    myFile.write((char *)(&num_of_triangles), 4);

    PrintError("num_of_triangles : %d", num_of_triangles);

    ResetConsoleProgress(num_of_triangles, "Writing STL: ");
    for (int i = 0; i < num_of_triangles; i++) {
        char char_buffer[12];
        char blank[2] = {0, 0};

        Eigen::Vector3f float_vector3f =
            TriangleMesh.triangle_normals_[i].cast<float>();
        myFile.write(reinterpret_cast<const char *>(float_vector3f.data()), 12);
        for (int j = 0; j < 3; j++) {
            Eigen::Vector3f float_vector3f =
                TriangleMesh.vertices_[i * 3 + j].cast<float>();
            myFile.write(reinterpret_cast<const char *>(float_vector3f.data()),
                         12);
        }
        myFile.write(blank, 2);
        AdvanceConsoleProgress();
    }

    return true;
}

}  // namespace open3d
