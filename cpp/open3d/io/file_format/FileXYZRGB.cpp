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

#include "open3d/io/FileFormatIO.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace io {

FileGeometry ReadFileGeometryTypeXYZRGB(const std::string &path) {
    return CONTAINS_POINTS;
}

bool ReadPointCloudFromXYZRGB(const std::string &filename,
                              geometry::PointCloud &pointcloud,
                              const ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZRGB failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());

        pointcloud.Clear();
        int i = 0;
        double x, y, z, r, g, b;
        const char *line_buffer;
        while ((line_buffer = file.ReadLine())) {
            if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &r,
                       &g, &b) == 6) {
                pointcloud.points_.push_back(Eigen::Vector3d(x, y, z));
                pointcloud.colors_.push_back(Eigen::Vector3d(r, g, b));
            }
            if (++i % 1000 == 0) {
                reporter.Update(file.CurPos());
            }
        }
        reporter.Finish();

        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToXYZRGB(const std::string &filename,
                             const geometry::PointCloud &pointcloud,
                             const WritePointCloudOption &params) {
    if (!pointcloud.HasColors()) {
        return false;
    }

    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZRGB failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(pointcloud.points_.size());

        for (size_t i = 0; i < pointcloud.points_.size(); i++) {
            const Eigen::Vector3d &point = pointcloud.points_[i];
            const Eigen::Vector3d &color = pointcloud.colors_[i];
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                        point(0), point(1), point(2), color(0), color(1),
                        color(2)) < 0) {
                utility::LogWarning(
                        "Write XYZRGB failed: unable to write file: {}",
                        filename);
                return false;  // error happened during writing.
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write XYZRGB failed with exception: {}", e.what());
        return false;
    }
}

}  // namespace io
}  // namespace open3d
