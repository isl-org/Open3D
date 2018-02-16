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

#include <IO/ClassIO/PinholeCameraTrajectoryIO.h>

#include <Eigen/Dense>
#include <Core/Utility/Console.h>

// The log file is the redwood-data format for camera trajectories
// See these pages for details:
// http://redwood-data.org/indoor/fileformat.html
// https://github.com/qianyizh/ElasticReconstruction/blob/f986e81a46201e28c0408a5f6303b4d3cdac7423/GraphOptimizer/helper.h

namespace three{

bool ReadPinholeCameraTrajectoryFromLOG(const std::string &filename,
		PinholeCameraTrajectory &trajectory)
{
	if (trajectory.intrinsic_.IsValid() == false) {
		trajectory.intrinsic_ = PinholeCameraIntrinsic::PrimeSenseDefault;
	}
	trajectory.extrinsic_.clear();
	FILE * f = fopen(filename.c_str(), "r");
	if (f == NULL) {
		PrintWarning("Read LOG failed: unable to open file: %s\n", filename.c_str());
		return false;
	}
	char line_buffer[DEFAULT_IO_BUFFER_SIZE];
	int i, j, k;
	Eigen::Matrix4d trans;
	while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
		if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
			if (sscanf(line_buffer, "%d %d %d", &i, &j, &k) != 3) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
				return false;
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(0,0), &trans(0,1),
						&trans(0,2), &trans(0,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(1,0), &trans(1,1),
						&trans(1,2), &trans(1,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(2,0), &trans(2,1),
						&trans(2,2), &trans(2,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
                fclose(f);
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(3,0), &trans(3,1),
						&trans(3,2), &trans(3,3));
			}
			trajectory.extrinsic_.push_back(trans.inverse());
		}
	}
	fclose(f);
	return true;
}

bool WritePinholeCameraTrajectoryToLOG(const std::string &filename,
		const PinholeCameraTrajectory &trajectory)
{
	FILE * f = fopen( filename.c_str(), "w" );
	if (f == NULL) {
		PrintWarning("Write LOG failed: unable to open file: %s\n", filename.c_str());
		return false;
	}
	for (size_t i = 0; i < trajectory.extrinsic_.size(); i++ ) {
		const auto &trans = trajectory.extrinsic_[i];
		fprintf(f, "%d %d %d\n", (int)i, (int)i, (int)i + 1);
		fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(0,0), trans(0,1), trans(0,2),
				trans(0,3) );
		fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(1,0), trans(1,1), trans(1,2),
				trans(1,3) );
		fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(2,0), trans(2,1), trans(2,2),
				trans(2,3) );
		fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(3,0), trans(3,1), trans(3,2),
				trans(3,3) );
	}
	fclose( f );
	return true;
}

}	// namespace three
