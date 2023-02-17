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

#include "open3d/core/TensorCheck.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace t {
namespace io {

bool ReadPointCloudFromPTS(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           const ReadPointCloudOption &params) {
    try {
        // Pointcloud is empty if the file is not read successfully.
        pointcloud.Clear();

        // Get num_points.
        utility::filesystem::CFile file;
        if (!file.Open(filename, "rb")) {
            utility::LogWarning("Read PTS failed: unable to open file: {}",
                                filename);
            return false;
        }

        int64_t num_points = 0;
        const char *line_buffer;
        if ((line_buffer = file.ReadLine())) {
            const char *format;
            if (std::is_same<int64_t, long>::value) {
                format = "%ld";
            } else {
                format = "%lld";
            }
            sscanf(line_buffer, format, &num_points);
        }
        if (num_points < 0) {
            utility::LogWarning(
                    "Read PTS failed: number of points must be >= 0.");
            return false;
        } else if (num_points == 0) {
            pointcloud.SetPointPositions(core::Tensor({0, 3}, core::Float32));
            return true;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(num_points);

        // Store data start position.
        int64_t start_pos = ftell(file.GetFILE());

        float *points_ptr = nullptr;
        float *intensities_ptr = nullptr;
        uint8_t *colors_ptr = nullptr;
        size_t num_fields = 0;

        if ((line_buffer = file.ReadLine())) {
            num_fields = utility::SplitString(line_buffer, " ").size();

            // X Y Z I R G B.
            if (num_fields == 7) {
                pointcloud.SetPointPositions(
                        core::Tensor({num_points, 3}, core::Float32));
                points_ptr = pointcloud.GetPointPositions().GetDataPtr<float>();
                pointcloud.SetPointAttr(
                        "intensities",
                        core::Tensor({num_points, 1}, core::Float32));
                intensities_ptr = pointcloud.GetPointAttr("intensities")
                                          .GetDataPtr<float>();
                pointcloud.SetPointColors(
                        core::Tensor({num_points, 3}, core::UInt8));
                colors_ptr = pointcloud.GetPointColors().GetDataPtr<uint8_t>();
            }
            // X Y Z R G B.
            else if (num_fields == 6) {
                pointcloud.SetPointPositions(
                        core::Tensor({num_points, 3}, core::Float32));
                points_ptr = pointcloud.GetPointPositions().GetDataPtr<float>();
                pointcloud.SetPointColors(
                        core::Tensor({num_points, 3}, core::UInt8));
                colors_ptr = pointcloud.GetPointColors().GetDataPtr<uint8_t>();
            }
            // X Y Z I.
            else if (num_fields == 4) {
                pointcloud.SetPointPositions(
                        core::Tensor({num_points, 3}, core::Float32));
                points_ptr = pointcloud.GetPointPositions().GetDataPtr<float>();
                pointcloud.SetPointAttr(
                        "intensities",
                        core::Tensor({num_points, 1}, core::Float32));
                intensities_ptr = pointcloud.GetPointAttr("intensities")
                                          .GetDataPtr<float>();
            }
            // X Y Z.
            else if (num_fields == 3) {
                pointcloud.SetPointPositions(
                        core::Tensor({num_points, 3}, core::Float32));
                points_ptr = pointcloud.GetPointPositions().GetDataPtr<float>();
            } else {
                utility::LogWarning("Read PTS failed: unknown pts format: {}",
                                    line_buffer);
                return false;
            }
        }

        // Go to data start position.
        fseek(file.GetFILE(), start_pos, 0);

        int64_t idx = 0;
        while (idx < num_points && (line_buffer = file.ReadLine())) {
            float x, y, z, i;
            int r, g, b;
            // X Y Z I R G B.
            if (num_fields == 7 && (sscanf(line_buffer, "%f %f %f %f %d %d %d",
                                           &x, &y, &z, &i, &r, &g, &b) == 7)) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                intensities_ptr[idx] = i;
                colors_ptr[3 * idx + 0] = r;
                colors_ptr[3 * idx + 1] = g;
                colors_ptr[3 * idx + 2] = b;
            }
            // X Y Z R G B.
            else if (num_fields == 6 &&
                     (sscanf(line_buffer, "%f %f %f %d %d %d", &x, &y, &z, &r,
                             &g, &b) == 6)) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                colors_ptr[3 * idx + 0] = r;
                colors_ptr[3 * idx + 1] = g;
                colors_ptr[3 * idx + 2] = b;
            }
            // X Y Z I.
            else if (num_fields == 4 && (sscanf(line_buffer, "%f %f %f %f", &x,
                                                &y, &z, &i) == 4)) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
                intensities_ptr[idx] = i;
            }
            // X Y Z.
            else if (num_fields == 3 &&
                     sscanf(line_buffer, "%f %f %f", &x, &y, &z) == 3) {
                points_ptr[3 * idx + 0] = x;
                points_ptr[3 * idx + 1] = y;
                points_ptr[3 * idx + 2] = z;
            } else {
                utility::LogWarning("Read PTS failed at line: {}", line_buffer);
                return false;
            }
            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        }

        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read PTS failed with exception: {}", e.what());
        return false;
    }
}

core::Tensor ConvertColorTensorToUint8(const core::Tensor &color_in) {
    core::Tensor color_values;
    if (color_in.GetDtype() == core::Float32 ||
        color_in.GetDtype() == core::Float64) {
        color_values = color_in.Clip(0, 1).Mul(255).Round();
    } else if (color_in.GetDtype() == core::Bool) {
        color_values = color_in.To(core::UInt8) * 255;
    } else {
        color_values = color_in;
    }
    return color_values.To(core::UInt8);
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

        utility::CountingProgressReporter reporter(params.update_progress);
        int64_t num_points = 0;

        if (!pointcloud.IsEmpty()) {
            num_points = pointcloud.GetPointPositions().GetLength();
        }

        // Assert attribute shapes.
        if (pointcloud.HasPointPositions()) {
            core::AssertTensorShape(pointcloud.GetPointPositions(),
                                    {num_points, 3});
        }
        if (pointcloud.HasPointColors()) {
            core::AssertTensorShape(pointcloud.GetPointColors(),
                                    {num_points, 3});
        }
        if (pointcloud.HasPointAttr("intensities")) {
            core::AssertTensorShape(pointcloud.GetPointAttr("intensities"),
                                    {num_points, 1});
        }

        reporter.SetTotal(num_points);

        if (fprintf(file.GetFILE(), "%zu\r\n", (size_t)num_points) < 0) {
            utility::LogWarning("Write PTS failed: unable to write file: {}",
                                filename);
            return false;
        }

        const float *points_ptr = nullptr;
        const float *intensities_ptr = nullptr;
        const uint8_t *colors_ptr = nullptr;
        core::Tensor colors;

        if (num_points > 0) {
            points_ptr = pointcloud.GetPointPositions()
                                 .To(core::Float32)
                                 .GetDataPtr<float>();
        }

        // X Y Z I R G B.
        if (pointcloud.HasPointColors() &&
            pointcloud.HasPointAttr("intensities")) {
            colors = ConvertColorTensorToUint8(pointcloud.GetPointColors());
            colors_ptr = colors.GetDataPtr<uint8_t>();
            intensities_ptr = pointcloud.GetPointAttr("intensities")
                                      .To(core::Float32)
                                      .GetDataPtr<float>();
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(),
                            "%.10f %.10f %.10f %.10f %d %d %d\r\n",
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
        }
        // X Y Z R G B.
        else if (pointcloud.HasPointColors()) {
            colors = ConvertColorTensorToUint8(pointcloud.GetPointColors());
            colors_ptr = colors.GetDataPtr<uint8_t>();
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %d %d %d\r\n",
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
        }
        // X Y Z I.
        else if (pointcloud.HasPointAttr("intensities")) {
            intensities_ptr = pointcloud.GetPointAttr("intensities")
                                      .To(core::Float32)
                                      .GetDataPtr<float>();
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
        }
        // X Y Z.
        else {
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
