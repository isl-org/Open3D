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

#include <cstdio>

#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

namespace open3d {
namespace io {

bool ReadPointCloudFromXYZN(const std::string &filename,
                            geometry::PointCloud &pointcloud,
                            bool print_progress) {
    FILE *file = utility::filesystem::FOpen(filename, "r");
    if (file == NULL) {
        utility::LogWarning("Read XYZN failed: unable to open file: {}",
                            filename);
        return false;
    }

    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    double x, y, z, nx, ny, nz;
    pointcloud.Clear();

    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &nx, &ny,
                   &nz) == 6) {
            pointcloud.points_.push_back(Eigen::Vector3d(x, y, z));
            pointcloud.normals_.push_back(Eigen::Vector3d(nx, ny, nz));
        }
    }

    fclose(file);
    return true;
}

bool WritePointCloudToXYZN(const std::string &filename,
                           const geometry::PointCloud &pointcloud,
                           bool write_ascii /* = false*/,
                           bool compressed /* = false*/,
                           bool print_progress) {
    if (pointcloud.HasNormals() == false) {
        return false;
    }

    FILE *file = utility::filesystem::FOpen(filename, "w");
    if (file == NULL) {
        utility::LogWarning("Write XYZN failed: unable to open file: {}",
                            filename);
        return false;
    }

    for (size_t i = 0; i < pointcloud.points_.size(); i++) {
        const Eigen::Vector3d &point = pointcloud.points_[i];
        const Eigen::Vector3d &normal = pointcloud.normals_[i];
        if (fprintf(file, "%.10f %.10f %.10f %.10f %.10f %.10f\n", point(0),
                    point(1), point(2), normal(0), normal(1), normal(2)) < 0) {
            utility::LogWarning("Write XYZN failed: unable to write file: {}",
                                filename);
            fclose(file);
            return false;  // error happens during writing.
        }
    }

    fclose(file);
    return true;
}

}  // namespace io
}  // namespace open3d
