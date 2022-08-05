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

bool ReadFileTXT(const std::string &filename,
                 geometry::PointCloud &pointcloud,
                 const std::vector<std::string> &line_template,
                 const open3d::io::ReadPointCloudOption &params) {
    utility::filesystem::CFile file;
    if (!file.Open(filename, "r")) {
        utility::LogWarning("Read TXT failed: unable to open file: {}",
                            filename);
        return false;
    }
    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(file.GetFileSize());

    int64_t num_points = file.GetNumLines();
    int64_t num_elements_per_line = line_template.size();

    pointcloud.Clear();
    core::Tensor pcd_buffer({num_points, num_elements_per_line}, core::Float64);
    double *pcd_buffer_ptr = pcd_buffer.GetDataPtr<double>();

    int i = 0;
    double x, y, z, attr, attr_x, attr_y, attr_z;
    const char *line_buffer;

    // Read TXT to buffer
    while ((line_buffer = file.ReadLine())) {
        if (num_elements_per_line == 3 &&
            (sscanf(line_buffer, "%lf %lf %lf", &x, &y, &z) == 3)) {
            pcd_buffer_ptr[num_elements_per_line * i + 0] = x;
            pcd_buffer_ptr[num_elements_per_line * i + 1] = y;
            pcd_buffer_ptr[num_elements_per_line * i + 2] = z;
        } else if (num_elements_per_line == 4 &&
                   (sscanf(line_buffer, "%lf %lf %lf %lf", &x, &y, &z, &attr) ==
                    4)) {
            pcd_buffer_ptr[num_elements_per_line * i + 0] = x;
            pcd_buffer_ptr[num_elements_per_line * i + 1] = y;
            pcd_buffer_ptr[num_elements_per_line * i + 2] = z;
            pcd_buffer_ptr[num_elements_per_line * i + 3] = attr;
        } else if (num_elements_per_line == 6 &&
                   (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf", &x, &y, &z,
                           &attr_x, &attr_y, &attr_z) == 6)) {
            pcd_buffer_ptr[num_elements_per_line * i + 0] = x;
            pcd_buffer_ptr[num_elements_per_line * i + 1] = y;
            pcd_buffer_ptr[num_elements_per_line * i + 2] = z;
            pcd_buffer_ptr[num_elements_per_line * i + 3] = attr_x;
            pcd_buffer_ptr[num_elements_per_line * i + 4] = attr_y;
            pcd_buffer_ptr[num_elements_per_line * i + 5] = attr_z;
        } else {
            utility::LogWarning("Read TXT failed at line: {}", line_buffer);
            return false;
        }
        if (++i % 1000 == 0) {
            reporter.Update(file.CurPos());
        }
    }

    // Buffer to point cloud
    pointcloud.SetPointPositions(pcd_buffer.Slice(1, 0, 3, 1));
    if (line_template == std::vector<std::string>{"x", "y", "z"})
        ;
    else if (line_template == std::vector<std::string>{"x", "y", "z", "i"}) {
        pointcloud.SetPointAttr("intensities", pcd_buffer.Slice(1, 3, 4, 1));
    } else if (line_template ==
               std::vector<std::string>{"x", "y", "z", "nx", "ny", "nz"}) {
        pointcloud.SetPointAttr("normals", pcd_buffer.Slice(1, 3, 6, 1));
    } else if (line_template ==
               std::vector<std::string>{"x", "y", "z", "r", "g", "b"}) {
        pointcloud.SetPointAttr("colors", pcd_buffer.Slice(1, 3, 6, 1));
    } else {
        utility::LogWarning("The format of TXT is not supported");
        return false;
    }
    reporter.Finish();
    return true;
}

bool ReadPointCloudFromTXT(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           const open3d::io::ReadPointCloudOption &params) {
    try {
        std::string formatTXT =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
        if (formatTXT == "xyz") {
            return ReadFileTXT(filename, pointcloud,
                               std::vector<std::string>{"x", "y", "z"}, params);
        } else if (formatTXT == "xyzi") {
            return ReadFileTXT(filename, pointcloud,
                               std::vector<std::string>{"x", "y", "z", "i"},
                               params);
        } else if (formatTXT == "xyzn") {
            return ReadFileTXT(
                    filename, pointcloud,
                    std::vector<std::string>{"x", "y", "z", "nx", "ny", "nz"},
                    params);
        } else if (formatTXT == "xyzrgb") {
            return ReadFileTXT(
                    filename, pointcloud,
                    std::vector<std::string>{"x", "y", "z", "r", "g", "b"},
                    params);
        } else {
            utility::LogWarning("The format of TXT is not supported");
            return false;
        }

    } catch (const std::exception &e) {
        utility::LogWarning("Read TXT failed with exception: {}", e.what());
        return false;
    }
}

bool WriteFileTXT(const std::string &filename,
                  const geometry::PointCloud &pointcloud,
                  const std::vector<std::string> &line_template,
                  const open3d::io::WritePointCloudOption &params) {
    utility::filesystem::CFile file;
    if (!file.Open(filename, "w")) {
        utility::LogWarning("Write TXT failed: unable to open file: {}",
                            filename);
        return false;
    }
    utility::CountingProgressReporter reporter(params.update_progress);
    const core::Tensor &points = pointcloud.GetPointPositions();

    if (!points.GetShape().IsCompatible({utility::nullopt, 3})) {
        utility::LogWarning(
                "Write TXT failed: Shape of points is {}, but it should "
                "be Nx3.",
                points.GetShape());
        return false;
    }

    reporter.SetTotal(points.GetShape(0));

    if (line_template == std::vector<std::string>{"x", "y", "z"}) {
        for (int i = 0; i < points.GetShape(0); i++) {
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f\n",
                        points[i][0].Item<double>(),
                        points[i][1].Item<double>(),
                        points[i][2].Item<double>()) < 0) {
                utility::LogWarning(
                        "Write TXT failed: unable to write file: {}", filename);
                return false;  // error happened during writing.
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
    } else {
        if (line_template == std::vector<std::string>{"x", "y", "z", "i"}) {
            const core::Tensor &intensities =
                    pointcloud.GetPointAttr("intensities");
            if (points.GetShape(0) != intensities.GetShape(0)) {
                utility::LogWarning(
                        "Write TXT failed: Points ({}) and intensities ({}) "
                        "have "
                        "different lengths.",
                        points.GetShape(0), intensities.GetShape(0));
                return false;
            }

            for (int i = 0; i < points.GetShape(0); i++) {
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f\n",
                            points[i][0].Item<double>(),
                            points[i][1].Item<double>(),
                            points[i][2].Item<double>(),
                            intensities[i][0].Item<double>()) < 0) {
                    utility::LogWarning(
                            "Write TXT failed: unable to write file: {}",
                            filename);
                    return false;  // error happened during writing.
                }
                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else if (line_template ==
                   std::vector<std::string>{"x", "y", "z", "nx", "ny", "nz"}) {
            const core::Tensor &normals = pointcloud.GetPointAttr("normals");
            if (points.GetShape(0) != normals.GetShape(0)) {
                utility::LogWarning(
                        "Write TXT failed: Points ({}) and normals ({}) "
                        "have "
                        "different lengths.",
                        points.GetShape(0), normals.GetShape(0));
                return false;
            }

            for (int i = 0; i < points.GetShape(0); i++) {
                if (fprintf(file.GetFILE(),
                            "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                            points[i][0].Item<double>(),
                            points[i][1].Item<double>(),
                            points[i][2].Item<double>(),
                            normals[i][0].Item<double>(),
                            normals[i][1].Item<double>(),
                            normals[i][2].Item<double>()) < 0) {
                    utility::LogWarning(
                            "Write TXT failed: unable to write file: {}",
                            filename);
                    return false;  // error happened during writing.
                }
                if (i % 1000 == 0) {
                    reporter.Update(i);
                }
            }
        } else if (line_template ==
                   std::vector<std::string>{"x", "y", "z", "r", "g", "b"}) {
            const core::Tensor &colors = pointcloud.GetPointAttr("colors");

            if (points.GetShape(0) != colors.GetShape(0)) {
                utility::LogWarning(
                        "Write TXT failed: Points ({}) and colors ({}) "
                        "have "
                        "different lengths.",
                        points.GetShape(0), colors.GetShape(0));
                return false;
            }

            for (int i = 0; i < points.GetShape(0); i++) {
                if (fprintf(file.GetFILE(),
                            "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                            points[i][0].Item<double>(),
                            points[i][1].Item<double>(),
                            points[i][2].Item<double>(),
                            colors[i][0].Item<double>(),
                            colors[i][1].Item<double>(),
                            colors[i][2].Item<double>()) < 0) {
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
            utility::LogWarning("The format of TXT is not supported");
            return false;
        }
    }

    reporter.Finish();
    return true;
}

bool WritePointCloudToTXT(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          const open3d::io::WritePointCloudOption &params) {
    try {
        std::string formatTXT =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
        if (formatTXT == "xyz") {
            return WriteFileTXT(filename, pointcloud,
                                std::vector<std::string>{"x", "y", "z"},
                                params);
        } else if (formatTXT == "xyzi") {
            return WriteFileTXT(filename, pointcloud,
                                std::vector<std::string>{"x", "y", "z", "i"},
                                params);
        } else if (formatTXT == "xyzn") {
            return WriteFileTXT(
                    filename, pointcloud,
                    std::vector<std::string>{"x", "y", "z", "nx", "ny", "nz"},
                    params);
        } else if (formatTXT == "xyzrgb") {
            return WriteFileTXT(
                    filename, pointcloud,
                    std::vector<std::string>{"x", "y", "z", "r", "g", "b"},
                    params);
        } else {
            utility::LogWarning("The format of TXT is not supported");
            return false;
        }

    } catch (const std::exception &e) {
        utility::LogWarning("Write TXT failed with exception: {}", e.what());
        return false;
    }
}

}  // namespace io
}  // namespace t
}  // namespace open3d
