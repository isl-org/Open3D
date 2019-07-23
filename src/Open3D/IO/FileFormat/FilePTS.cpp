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
#include "Open3D/Utility/Helper.h"

namespace open3d {
namespace io {

bool ReadPointCloudFromPTS(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           bool print_progress) {
    FILE *file = fopen(filename.c_str(), "r");
    if (file == NULL) {
        utility::LogWarning("Read PTS failed: unable to open file.\n");
        return false;
    }
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    int num_of_pts = 0, num_of_fields = 0;
    if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        sscanf(line_buffer, "%d", &num_of_pts);
    }
    if (num_of_pts <= 0) {
        utility::LogWarning("Read PTS failed: unable to read header.\n");
        fclose(file);
        return false;
    }
    utility::ConsoleProgressBar progress_bar(num_of_pts,
                                             "Reading PTS: ", print_progress);
    int idx = 0;
    while (idx < num_of_pts &&
           fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        if (num_of_fields == 0) {
            std::vector<std::string> st;
            utility::SplitString(st, line_buffer, " ");
            num_of_fields = (int)st.size();
            if (num_of_fields < 3) {
                utility::LogWarning(
                        "Read PTS failed: insufficient data fields.\n");
                fclose(file);
                return false;
            }
            pointcloud.points_.resize(num_of_pts);
            if (num_of_fields >= 7) {
                // X Y Z I R G B
                pointcloud.colors_.resize(num_of_pts);
            }
        }
        double x, y, z;
        int i, r, g, b;
        if (num_of_fields < 7) {
            if (sscanf(line_buffer, "%lf %lf %lf", &x, &y, &z) == 3) {
                pointcloud.points_[idx] = Eigen::Vector3d(x, y, z);
            }
        } else {
            if (sscanf(line_buffer, "%lf %lf %lf %d %d %d %d", &x, &y, &z, &i,
                       &r, &g, &b) == 7) {
                pointcloud.points_[idx] = Eigen::Vector3d(x, y, z);
                pointcloud.colors_[idx] = Eigen::Vector3d(r, g, b) / 255.0;
            }
        }
        idx++;
        ++progress_bar;
    }
    fclose(file);
    return true;
}

bool WritePointCloudToPTS(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          bool write_ascii /* = false*/,
                          bool compressed /* = false*/,
                          bool print_progress) {
    FILE *file = fopen(filename.c_str(), "w");
    if (file == NULL) {
        utility::LogWarning("Write PTS failed: unable to open file.\n");
        return false;
    }
    fprintf(file, "%d\r\n", (int)pointcloud.points_.size());
    utility::ConsoleProgressBar progress_bar(
            static_cast<size_t>(pointcloud.points_.size()),
            "Writing PTS: ", print_progress);
    for (size_t i = 0; i < pointcloud.points_.size(); i++) {
        const auto &point = pointcloud.points_[i];
        if (pointcloud.HasColors() == false) {
            fprintf(file, "%.10f %.10f %.10f\r\n", point(0), point(1),
                    point(2));
        } else {
            const auto &color = pointcloud.colors_[i] * 255.0;
            fprintf(file, "%.10f %.10f %.10f %d %d %d %d\r\n", point(0),
                    point(1), point(2), 0, (int)color(0), (int)color(1),
                    (int)(color(2)));
        }
        ++progress_bar;
    }
    fclose(file);
    return true;
}

}  // namespace io
}  // namespace open3d
