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
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace t {
namespace io {

bool ReadPointCloudFromPTS(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           const ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read PTS failed: unable to open file: {}",
                                filename);
            return false;
        }

        size_t num_pts = 0;
        const char *line_buffer;
        if ((line_buffer = file.ReadLine())) {
            sscanf(line_buffer, "%zu", &num_pts);
        }
        if (static_cast<int64_t>(num_pts) <= 0) {
            utility::LogWarning("Read PTS failed: Number of points <= 0");
            return false;
        }

        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(num_pts);
        pointcloud.Clear();

        double *points_ptr = nullptr;
        double *intensities_ptr = nullptr;
        uint8_t *colors_ptr = nullptr;
        size_t idx = 0;
        std::vector<std::string> st;
        int num_fields = 0;

        if ((line_buffer = file.ReadLine())) {
            st = utility::SplitString(line_buffer, " ");
            num_fields = static_cast<int64_t>(st.size());
            if (num_fields < 3) {
                utility::LogWarning(
                        "Read PTS failed: insufficient data fields.");
                return false;
            }

            pointcloud.SetPoints(core::Tensor(
                    {static_cast<int64_t>(num_pts), 3}, core::Dtype::Float64));
            points_ptr = pointcloud.GetPoints().GetDataPtr<double>();

            // X Y Z I or X Y Z I R G B
            if (num_fields == 4 || num_fields == 7) {
                pointcloud.SetPointAttr(
                        "intensities",
                        core::Tensor({static_cast<int64_t>(num_pts), 1},
                                     core::Dtype::Float64));
                intensities_ptr = pointcloud.GetPointAttr("intensities")
                                          .GetDataPtr<double>();
            }

            // X Y Z R G B or X Y Z I R G B
            if (num_fields == 6 || num_fields == 7) {
                pointcloud.SetPointColors(
                        core::Tensor({static_cast<int64_t>(num_pts), 3},
                                     core::Dtype::UInt8));
                colors_ptr = pointcloud.GetPointColors().GetDataPtr<uint8_t>();
            }
        }

        do {
            st.clear();
            st = utility::SplitString(line_buffer, " ");
            if (num_fields > (int)st.size()) {
                utility::LogWarning(
                        "Read PTS failed: lines have unequal elements.");
                return false;
            }

            double x, y, z, i;
            int r, g, b;
            if (num_fields == 7 &&
                (sscanf(line_buffer, "%lf %lf %lf %lf %d %d %d", &x, &y, &z, &i,
                        &r, &g, &b) == 7)) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                intensities_ptr[idx] = i;
                colors_ptr[3 * idx + 0] = r;
                colors_ptr[3 * idx + 1] = g;
                colors_ptr[3 * idx + 2] = b;
            } else if (num_fields == 6 &&
                       (sscanf(line_buffer, "%lf %lf %lf %d %d %d", &x, &y, &z,
                               &r, &g, &b) == 6)) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                colors_ptr[3 * idx + 0] = r;
                colors_ptr[3 * idx + 1] = g;
                colors_ptr[3 * idx + 2] = b;
            } else if (num_fields == 4 &&
                       (sscanf(line_buffer, "%lf %lf %lf %lf", &x, &y, &z,
                               &i) == 4)) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                intensities_ptr[idx] = i;
            } else if (num_fields == 3 &&
                       sscanf(line_buffer, "%lf %lf %lf", &x, &y, &z) == 3) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
            } else {
                utility::LogWarning("Read PTS failed: unknown pts format");
                return false;
            }
            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        } while (idx < num_pts && (line_buffer = file.ReadLine()));

        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read PTS failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToPTS(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          const WritePointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write PTS failed: unable to open file: {}",
                                filename);
            return false;
        }

        if (pointcloud.IsEmpty()) {
            utility::LogWarning("Write PTS failed: point cloud has 0 points.");
            return false;
        }

        utility::CountingProgressReporter reporter(params.update_progress);
        int64_t num_points =
                static_cast<long>(pointcloud.GetPoints().GetLength());
        const double *points_ptr = static_cast<const double *>(
                pointcloud.GetPoints().GetDataPtr());
        const uint8_t *colors_ptr = nullptr;
        const double *intensities_ptr = nullptr;

        if (pointcloud.HasPointColors()) {
            colors_ptr = static_cast<const uint8_t *>(
                    pointcloud.GetPointColors().GetDataPtr());
        }

        if (pointcloud.HasPointAttr("intensities")) {
            intensities_ptr = static_cast<const double *>(
                    pointcloud.GetPointAttr("intensities")
                            .GetDataPtr<double>());
        }

        reporter.SetTotal(num_points);

        if (fprintf(file.GetFILE(), "%zu\r\n", (size_t)num_points) < 0) {
            utility::LogWarning("Write PTS failed: unable to write file: {}",
                                filename);
            return false;
        }

        if (pointcloud.HasPointColors() &&
            pointcloud.HasPointAttr("intensities")) {
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(),
                            "%.10f %.10f %.10f %.10f %u %u %u\r\n",
                            points_ptr[3 * i + 0], points_ptr[3 * i + 1],
                            points_ptr[3 * i + 2], intensities_ptr[i],
                            colors_ptr[3 * i + 0], colors_ptr[3 * i + 1],
                            colors_ptr[3 * i + 2]) < 0) {
                    utility::LogWarning(
                            "Write PTS failed: unable to write file: {}",
                            filename);
                    return false;
                }

                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else if (pointcloud.HasPointColors()) {
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %u %u %u\r\n",
                            points_ptr[3 * i + 0], points_ptr[3 * i + 1],
                            points_ptr[3 * i + 2], colors_ptr[3 * i + 0],
                            colors_ptr[3 * i + 1], colors_ptr[3 * i + 2]) < 0) {
                    utility::LogWarning(
                            "Write PTS failed: unable to write file: {}",
                            filename);
                    return false;
                }

                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else if (pointcloud.HasPointAttr("intensities")) {
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f\r\n",
                            points_ptr[3 * i + 0], points_ptr[3 * i + 1],
                            points_ptr[3 * i + 2], intensities_ptr[i]) < 0) {
                    utility::LogWarning(
                            "Write PTS failed: unable to write file: {}",
                            filename);
                    return false;
                }

                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else {
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f\r\n",
                            points_ptr[3 * i + 0], points_ptr[3 * i + 1],
                            points_ptr[3 * i + 2]) < 0) {
                    utility::LogWarning(
                            "Write PTS failed: unable to write file: {}",
                            filename);
                    return false;
                }

                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        }

        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write PTS failed with exception: {}", e.what());
        return false;
    }
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
