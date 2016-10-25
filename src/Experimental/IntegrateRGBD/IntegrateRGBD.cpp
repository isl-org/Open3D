// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include <iostream>
#include <memory>

#include <Core/Core.h>
#include <IO/IO.h>

int main(int argc, char *argv[])
{
	using namespace three;
	SetVerbosityLevel(VERBOSE_ALWAYS);
	
	if (argc < 3) {
		PrintInfo("Usage:\n");
		PrintInfo("    > IntegrateRGBD <match_file> <log_file>\n");
		return 0;
	}
	
	auto camera_trajectory = CreatePinholeCameraTrajectoryFromFile(argv[2]);
	std::string dir_name = filesystem::GetFileParentDirectory(argv[1]).c_str();
	FILE *file = fopen(argv[1], "r");
	if (file == NULL) {
		PrintError("Unable to open file %s\n", argv[1]);
		fclose(file);
		return 0;
	}
	char buffer[DEFAULT_IO_BUFFER_SIZE];
	int index = 0;
	FPSTimer timer("Process RGBD stream",
			(int)camera_trajectory->extrinsic_.size());
	while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
		std::vector<std::string> st;
		SplitString(st, buffer, "\t\r\n ");
		if (st.size() >= 2) {
			auto depth = CreateImageFromFile(dir_name + st[0]);
			auto image = CreateImageFromFile(dir_name + st[1]);
			auto camera_distance =
					CreateCameraDistanceFloatImageFromDepthImage(*depth,
					camera_trajectory->intrinsic_);
			timer.Signal();
		}
	}
	fclose(file);
	return 1;
}
