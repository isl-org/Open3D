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

open3d::io::FileGeometry ReadFileGeometryTypeTXT(const std::string &path) {
    return open3d::io::CONTAINS_POINTS;
}

bool ReadPointCloudFromTXT(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           const open3d::io::ReadPointCloudOption &params) {
    try {
        pointcloud.Clear();

        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read TXT failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());

        std::string format_txt =
                utility::filesystem::GetFileExtensionInLowerCase(filename);

        int64_t num_points = file.GetNumLines();
        int64_t num_attrs;

        if (format_txt == "xyz") {
            num_attrs = 3;
        } else if (format_txt == "xyzi") {
            num_attrs = 4;
        } else if (format_txt == "xyzn") {
            num_attrs = 6;
        } else if (format_txt == "xyzrgb") {
            num_attrs = 6;
        } else {
            utility::LogWarning("The format of {} is not supported",
                                format_txt);
            return false;
        }

        core::Tensor pcd_buffer({num_points, num_attrs}, core::Float32);
        float *pcd_buffer_ptr = pcd_buffer.GetDataPtr<float>();

        int i = 0;
        std::vector<float> line_attrs(num_attrs);
        float *line_attr_ptr = line_attrs.data();
        const char *line_buffer;

        // Read TXT to buffer.
        while ((line_buffer = file.ReadLine())) {
            if (num_attrs == 3 &&
                (sscanf(line_buffer, "%f %f %f", &line_attr_ptr[0],
                        &line_attr_ptr[1], &line_attr_ptr[2]) == 3)) {
                std::memcpy(&pcd_buffer_ptr[num_attrs * i], line_attr_ptr,
                            num_attrs * sizeof(float));
            } else if (num_attrs == 4 &&
                       (sscanf(line_buffer, "%f %f %f %f", &line_attr_ptr[0],
                               &line_attr_ptr[1], &line_attr_ptr[2],
                               &line_attr_ptr[3]) == 4)) {
                std::memcpy(&pcd_buffer_ptr[num_attrs * i], line_attr_ptr,
                            num_attrs * sizeof(float));
            } else if (num_attrs == 6 &&
                       (sscanf(line_buffer, "%f %f %f %f %f %f",
                               &line_attr_ptr[0], &line_attr_ptr[1],
                               &line_attr_ptr[2], &line_attr_ptr[3],
                               &line_attr_ptr[4], &line_attr_ptr[5]) == 6)) {
                std::memcpy(&pcd_buffer_ptr[num_attrs * i], line_attr_ptr,
                            num_attrs * sizeof(float));
            } else {
                utility::LogWarning("Read TXT failed at line: {}", line_buffer);
                return false;
            }
            if (++i % 1000 == 0) {
                reporter.Update(file.CurPos());
            }
        }

        // Buffer to point cloud.
        pointcloud.SetPointPositions(pcd_buffer.Slice(1, 0, 3, 1).Contiguous());
        if (format_txt == "xyz") {
            // No additional attributes.
        } else if (format_txt == "xyzi") {
            pointcloud.SetPointAttr("intensities",
                                    pcd_buffer.Slice(1, 3, 4, 1).Contiguous());
        } else if (format_txt == "xyzn") {
            pointcloud.SetPointAttr("normals",
                                    pcd_buffer.Slice(1, 3, 6, 1).Contiguous());
        } else if (format_txt == "xyzrgb") {
            pointcloud.SetPointAttr("colors",
                                    pcd_buffer.Slice(1, 3, 6, 1).Contiguous());
        } else {
            utility::LogWarning("The format of {} is not supported",
                                format_txt);
            return false;
        }
        reporter.Finish();
        return true;

    } catch (const std::exception &e) {
        utility::LogWarning("Read TXT failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToTXT(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          const open3d::io::WritePointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write TXT failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);

        std::string format_txt =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
        const core::Tensor &points = pointcloud.GetPointPositions();

        if (!points.GetShape().IsCompatible({utility::nullopt, 3})) {
            utility::LogWarning(
                    "Write TXT failed: Shape of points is {}, but it should "
                    "be Nx3.",
                    points.GetShape());
            return false;
        }
        reporter.SetTotal(points.GetShape(0));
        int64_t num_points = points.GetShape(0);
        int64_t len_attr;

        // Check attribute length.
        if (format_txt == "xyz") {
            // No additional attributes.
            len_attr = num_points;
        } else if (format_txt == "xyzi") {
            len_attr = pointcloud.GetPointAttr("intensities").GetLength();
        } else if (format_txt == "xyzn") {
            len_attr = pointcloud.GetPointAttr("normals").GetLength();
        } else if (format_txt == "xyzrgb") {
            len_attr = pointcloud.GetPointAttr("colors").GetLength();
        } else {
            utility::LogWarning("The format of {} is not supported",
                                format_txt);
            return false;
        }

        if (len_attr != num_points) {
            utility::LogWarning(
                    "Write TXT failed: Points ({}) and attributes ({}) have "
                    "different lengths.",
                    num_points, len_attr);
            return false;
        }

        const float *points_ptr = points.GetDataPtr<float>();
        const float *attr_ptr = nullptr;

        if (format_txt == "xyz") {
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f\n",
                            points_ptr[3 * i + 0], points_ptr[3 * i + 1],
                            points_ptr[3 * i + 2]) < 0) {
                    utility::LogWarning(
                            "Write TXT failed: unable to write file: {}",
                            filename);
                    return false;  // error happened during writing.
                }
                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else if (format_txt == "xyzi") {
            attr_ptr =
                    pointcloud.GetPointAttr("intensities").GetDataPtr<float>();
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f\n",
                            points_ptr[3 * i + 0], points_ptr[3 * i + 1],
                            points_ptr[3 * i + 2], attr_ptr[i]) < 0) {
                    utility::LogWarning(
                            "Write TXT failed: unable to write file: {}",
                            filename);
                    return false;  // error happened during writing.
                }
                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else if (format_txt == "xyzn") {
            attr_ptr = pointcloud.GetPointAttr("normals").GetDataPtr<float>();
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(),
                            "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                            points_ptr[3 * i + 0], points_ptr[3 * i + 1],
                            points_ptr[3 * i + 2], attr_ptr[3 * i + 0],
                            attr_ptr[3 * i + 1], attr_ptr[3 * i + 2]) < 0) {
                    utility::LogWarning(
                            "Write TXT failed: unable to write file: {}",
                            filename);
                    return false;  // error happened during writing.
                }
                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else if (format_txt == "xyzrgb") {
            attr_ptr = pointcloud.GetPointAttr("colors").GetDataPtr<float>();
            for (int i = 0; i < num_points; i++) {
                if (fprintf(file.GetFILE(),
                            "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                            points_ptr[3 * i + 0], points_ptr[3 * i + 1],
                            points_ptr[3 * i + 2], attr_ptr[3 * i + 0],
                            attr_ptr[3 * i + 1], attr_ptr[3 * i + 2]) < 0) {
                    utility::LogWarning(
                            "Write TXT failed: unable to write file: {}",
                            filename);
                    return false;  // error happened during writing.
                }
                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else {
            utility::LogWarning("The format of {} is not supported",
                                format_txt);
            return false;
        }

        reporter.Finish();
        return true;

    } catch (const std::exception &e) {
        utility::LogWarning("Write TXT failed with exception: {}", e.what());
        return false;
    }
}

}  // namespace io
}  // namespace t
}  // namespace open3d
