// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
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
#include <Visualization/Visualization.h>

int main(int argc, char *argv[])
{
	using namespace three;

	//SetVerbosityLevel(VERBOSE_ALWAYS);

	// 1. load some geometry
	
	std::shared_ptr<PointCloud> pointcloud_ptr(new PointCloud);
	if (ReadPointCloudFromPLY(argv[1], *pointcloud_ptr)) {
		PrintWarning("Successfully read %s\n", argv[1]);
	} else {
		PrintError("Failed to read %s\n\n", argv[1]);
	}
	pointcloud_ptr->NormalizeNormal();

	// 2. test visualization.

	Visualizer visualizer1;
	visualizer1.AddGeometry(pointcloud_ptr);
	visualizer1.CreateWindow("Open3DV", 1600, 900);

	Visualizer visualizer2;
	visualizer2.AddGeometry(pointcloud_ptr);
	visualizer2.CreateWindow("Open3DV", 800, 450);
	
	while (visualizer1.PollEvents() && visualizer2.PollEvents()) {
		
	}

	//while (!visualizer.IsWindowTerminated());
	// n. test end

	PrintInfo("End of the test.\n");
}