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
#include <thread>
#include <librealsense/rs.hpp>
#include <Core/Core.h>
#include <Visualization/Visualization.h>

using namespace three;

int main(int argc, char **args)
{
	rs::context ctx;
	PrintInfo("There are %d connected RealSense devices.\n",
			ctx.get_device_count());
	if(ctx.get_device_count() == 0) {
		return 0;
	}

	rs::device * dev = ctx.get_device(0);
	PrintInfo("Using device 0, an %s\n", dev->get_name());
	PrintInfo("    Serial number: %s\n", dev->get_serial());
	PrintInfo("    Firmware version: %s\n\n", dev->get_firmware_version());
	
	dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);
	dev->enable_stream(rs::stream::color, 1920, 1080, rs::format::rgb8, 30);
	dev->start();
	
	auto depth_image_ptr = std::make_shared<Image>();
	depth_image_ptr->PrepareImage(640, 480, 1, 2);
	auto color_image_ptr = std::make_shared<Image>();
	color_image_ptr->PrepareImage(1920, 1080, 3, 1);
	FPSTimer timer("Realsense stream");

	Visualizer depth_vis, color_vis;
	if (depth_vis.CreateWindow("Depth", 640, 480, 15, 50) == false ||
			depth_vis.AddGeometry(depth_image_ptr) == false ||
			color_vis.CreateWindow("Color", 1920, 1080, 675, 50) == false ||
			color_vis.AddGeometry(color_image_ptr) == false) {
		return 0;
	}
	
	while (depth_vis.PollEvents() && color_vis.PollEvents()) {
		timer.Signal();
		dev->wait_for_frames();
		memcpy(depth_image_ptr->data_.data(),
				dev->get_frame_data(rs::stream::depth), 640 * 480 * 2);
		memcpy(color_image_ptr->data_.data(),
				dev->get_frame_data(rs::stream::color), 1920 * 1080 * 3);
		depth_vis.UpdateGeometry();
		color_vis.UpdateGeometry();
	}
	
	//DrawGeometryWithAnimationCallback(depth_image_ptr,
	//		[&](Visualizer &vis) {
	//			timer.Signal();
	//			dev->wait_for_frames();
	//			memcpy(depth_image_ptr->data_.data(),
	//					dev->get_frame_data(rs::stream::depth), 640 * 480 * 2);
	//			return true;
	//		}, "Depth", 640, 480);
	return 1;
}