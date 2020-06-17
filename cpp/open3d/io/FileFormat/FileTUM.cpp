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
#include <Eigen/Geometry>

#include "Open3D/IO/ClassIO/PinholeCameraTrajectoryIO.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

// The TUM format for camera trajectories as used in
// "A Benchmark for the Evaluation of RGB-D SLAM Systems" by
// J. Sturm and N. Engelhard and F. Endres and W. Burgard and D. Cremers
// (IROS 2012)
// See these pages for details:
// https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
// https://vision.in.tum.de/data/datasets/rgbd-dataset

namespace open3d {
namespace io {

bool ReadPinholeCameraTrajectoryFromTUM(
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
    FILE *f = utility::filesystem::FOpen(filename, "r");
    if (f == NULL) {
        utility::LogWarning("Read TUM failed: unable to open file: {}",
                            filename);
        return false;
    }
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    double ts, x, y, z, qx, qy, qz, qw;
    Eigen::Matrix4d transform;
    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
        if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
            if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf %lf %lf", &ts, &x,
                       &y, &z, &qx, &qy, &qz, &qw) != 8) {
                utility::LogWarning("Read TUM failed: unrecognized format.");
                fclose(f);
                return false;
            }

            transform.setIdentity();
            transform.topLeftCorner<3, 3>() =
                    Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
            transform.topRightCorner<3, 1>() = Eigen::Vector3d(x, y, z);
            auto param = camera::PinholeCameraParameters();
            param.intrinsic_ = intrinsic;
            param.extrinsic_ = transform.inverse();
            trajectory.parameters_.push_back(param);
        }
    }
    fclose(f);
    return true;
}

bool WritePinholeCameraTrajectoryToTUM(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory) {
    FILE *f = utility::filesystem::FOpen(filename, "w");
    if (f == NULL) {
        utility::LogWarning("Write TUM failed: unable to open file: {}",
                            filename);
        return false;
    }

    Eigen::Quaterniond q;
    fprintf(f, "# TUM trajectory, format: <t> <x> <y> <z> <qx> <qy> <qz> <qw>");
    for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
        const Eigen::Matrix4d transform =
                trajectory.parameters_[i].extrinsic_.inverse();
        q = transform.topLeftCorner<3, 3>();
        fprintf(f, "%zu %lf %lf %lf %lf %lf %lf %lf\n", i, transform(0, 3),
                transform(1, 3), transform(2, 3), q.x(), q.y(), q.z(), q.w());
    }
    fclose(f);
    return true;
}

}  // namespace io
}  // namespace open3d
