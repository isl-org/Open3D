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

#include <Eigen/Dense>

#include "Open3D/IO/ClassIO/PinholeCameraTrajectoryIO.h"
#include "Open3D/Utility/Console.h"

// The log file is the redwood-data format for camera trajectories
// See these pages for details:
// http://redwood-data.org/indoor/fileformat.html
// https://github.com/qianyizh/ElasticReconstruction/blob/f986e81a46201e28c0408a5f6303b4d3cdac7423/GraphOptimizer/helper.h

namespace open3d {
namespace io {

bool ReadPinholeCameraTrajectoryFromLOG(
        const std::string &filename,
        camera::PinholeCameraTrajectory &trajectory) {
    camera::PinholeCameraIntrinsic intrinsic;
    if (trajectory.parameters_.size() >= 1 &&
        trajectory.parameters_[0].intrinsic_.IsValid()) {
        intrinsic = trajectory.parameters_[0].intrinsic_;
    } else {
        intrinsic = camera::PinholeCameraIntrinsic(
                camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    }
    trajectory.parameters_.clear();
    FILE *f = fopen(filename.c_str(), "r");
    if (f == NULL) {
        utility::LogWarning("Read LOG failed: unable to open file: {}\n",
                            filename.c_str());
        return false;
    }
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    int i, j, k;
    Eigen::Matrix4d trans;
    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
        if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
            if (sscanf(line_buffer, "%d %d %d", &i, &j, &k) != 3) {
                utility::LogWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
                return false;
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
                utility::LogWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(0, 0),
                       &trans(0, 1), &trans(0, 2), &trans(0, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
                utility::LogWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(1, 0),
                       &trans(1, 1), &trans(1, 2), &trans(1, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
                utility::LogWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(2, 0),
                       &trans(2, 1), &trans(2, 2), &trans(2, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
                utility::LogWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(3, 0),
                       &trans(3, 1), &trans(3, 2), &trans(3, 3));
            }
            auto param = camera::PinholeCameraParameters();
            param.intrinsic_ = intrinsic;
            param.extrinsic_ = trans.inverse();
            trajectory.parameters_.push_back(param);
        }
    }
    fclose(f);
    return true;
}

bool WritePinholeCameraTrajectoryToLOG(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory) {
    FILE *f = fopen(filename.c_str(), "w");
    if (f == NULL) {
        utility::LogWarning("Write LOG failed: unable to open file: {}\n",
                            filename);
        return false;
    }
    for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
        Eigen::Matrix4d_u trans =
                trajectory.parameters_[i].extrinsic_.inverse();
        fprintf(f, "%d %d %d\n", (int)i, (int)i, (int)i + 1);
        fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(0, 0), trans(0, 1),
                trans(0, 2), trans(0, 3));
        fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(1, 0), trans(1, 1),
                trans(1, 2), trans(1, 3));
        fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(2, 0), trans(2, 1),
                trans(2, 2), trans(2, 3));
        fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(3, 0), trans(3, 1),
                trans(3, 2), trans(3, 3));
    }
    fclose(f);
    return true;
}

}  // namespace io
}  // namespace open3d
