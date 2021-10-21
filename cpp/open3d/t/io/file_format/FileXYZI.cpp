// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace t {
namespace io {

open3d::io::FileGeometry ReadFileGeometryTypeXYZI(const std::string &path) {
    return open3d::io::CONTAINS_POINTS;
}

bool ReadPointCloudFromXYZI(const std::string &filename,
                            geometry::PointCloud &pointcloud,
                            const open3d::io::ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZI failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());
        int64_t num_points = file.GetNumLines();

        pointcloud.Clear();
        core::Tensor points({num_points, 3}, core::Float64);
        core::Tensor intensities({num_points, 1}, core::Float64);
        double *points_ptr = points.GetDataPtr<double>();
        double *intensities_ptr = intensities.GetDataPtr<double>();

        int i = 0;
        double x, y, z, I;
        const char *line_buffer;
        while ((line_buffer = file.ReadLine())) {
            if (sscanf(line_buffer, "%lf %lf %lf %lf", &x, &y, &z, &I) == 4) {
                points_ptr[3 * i + 0] = x;
                points_ptr[3 * i + 1] = y;
                points_ptr[3 * i + 2] = z;
                intensities_ptr[i] = I;
            }
            if (++i % 1000 == 0) {
                reporter.Update(file.CurPos());
            }
        }
        pointcloud.SetPointPositions(points);
        pointcloud.SetPointAttr("intensities", intensities);
        reporter.Finish();

        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToXYZI(const std::string &filename,
                           const geometry::PointCloud &pointcloud,
                           const open3d::io::WritePointCloudOption &params) {
    if (!pointcloud.HasPointAttr("intensities")) {
        return false;
    }

    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZI failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        const core::Tensor &points = pointcloud.GetPointPositions();
        if (!points.GetShape().IsCompatible({utility::nullopt, 3})) {
            utility::LogWarning(
                    "Write XYZI failed: Shape of points is {}, but it should "
                    "be Nx3.",
                    points.GetShape());
            return false;
        }
        const core::Tensor &intensities =
                pointcloud.GetPointAttr("intensities");
        if (points.GetShape(0) != intensities.GetShape(0)) {
            utility::LogWarning(
                    "Write XYZI failed: Points ({}) and intensities ({}) have "
                    "different lengths.",
                    points.GetShape(0), intensities.GetShape(0));
            return false;
        }
        reporter.SetTotal(points.GetShape(0));

        for (int i = 0; i < points.GetShape(0); i++) {
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f\n",
                        points[i][0].Item<double>(),
                        points[i][1].Item<double>(),
                        points[i][2].Item<double>(),
                        intensities[i][0].Item<double>()) < 0) {
                utility::LogWarning(
                        "Write XYZI failed: unable to write file: {}",
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
        utility::LogWarning("Write XYZI failed with exception: {}", e.what());
        return false;
    }
}

}  // namespace io
}  // namespace t
}  // namespace open3d
