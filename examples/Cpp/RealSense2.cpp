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

/**
    \file RealSense2.cpp
    \brief This file contains the RealSense2 example for using SDK 2 and cameras with Open3d.
*/
#include <iostream>
#include <thread>
#include "Open3D/Open3D.h"

using namespace open3d;

#include <imgui.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include "..\3rdparty\librealsense2\include\librealsense2\rs.hpp"  // Not sure how to set up includes using CMAKE.

/**
    Example program for SDK 2 RealSense devices.

    \param [in] argc Not used.
    \param [in] argv Not used.
*/
int main(int argc, char* argv[]) try {
    argc; // Reference to avoid C4100 compiler warning.
    argv[0]; // Reference to avoid C4100 compiler warning.

    rs2::colorizer c;  // Helper to colorize depth images

    /*
        Display information on all connected RealSense devices.
        See LibRealSense SDK 2 rs-sensor-control example for more detail.
    */
    rs2::context ctx;                                // The context represents the current platform with
                                                     // respect to connected devices.
    rs2::device_list devices = ctx.query_devices();  // Using the context, get all connected
                                                     // devices in a device list.

    if (0 == devices.size()) {
        std::cerr << "No device connected, please connect a RealSense device "
                     "and restart this program."
                  << std::endl;
        return 0;
    } else {
        std::cout << "Found the following devices:\n" << std::endl;
        int index = 0;
        for (rs2::device device : devices) {
            try {
                std::cout << "  " << index++ << " : " << device.get_info(RS2_CAMERA_INFO_NAME) << " : S/N "
                          << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << " Firmware "
                          << device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << std::endl;
            } catch (const rs2::error& e) {
                std::cerr << "Failed to get_info"
                          << ". (" << e.what() << ")" << std::endl;
            }
        }
    }
    /*
        Set same options as original RealSense SDK 1 example.
        D415 camera used for testing this development code.
    */
    rs2::device dev = devices[0];  // Use the first device.
    std::vector<rs2::sensor> sensors = dev.query_sensors();
    std::cout << "\nDevice consists of " << sensors.size() << " sensors:" << std::endl;
    int index = 0;
    // Iterate the sensors and print their names.
    for (rs2::sensor sensor : sensors) {
        if (sensor.supports(RS2_CAMERA_INFO_NAME) && sensor.supports(RS2_CAMERA_INFO_FIRMWARE_VERSION)) {
            std::cout << "  " << index++ << " : " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
        } else {
            std::cout << "  " << index++ << " : "
                      << "Unknown Sensor" << std::endl;
        }
    }
    rs2::sensor sensor = sensors[1];  // Use the RBG sensor for the device.

    /*
        Some options can only be set while the camera is streaming and generally
       the hardware might fail so it is good practice to catch exceptions from
       set_option.
    */
    if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
        try {
            sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.0);
        } catch (const rs2::error& e) {
            std::cerr << "Failed to set option " << RS2_OPTION_ENABLE_AUTO_EXPOSURE << ". (" << e.what() << ")" << std::endl;
        }
    } else {
        std::cerr << "RS2_OPTION_ENABLE_AUTO_EXPOSURE is not supported by this sensor" << std::endl;
    }
    if (sensor.supports(RS2_OPTION_EXPOSURE)) {
        try {
            sensor.set_option(RS2_OPTION_EXPOSURE, 625);
        } catch (const rs2::error& e) {
            std::cerr << "Failed to set option " << RS2_OPTION_EXPOSURE << ". (" << e.what() << ")" << std::endl;
        }
    } else {
        std::cerr << "RS2_OPTION_EXPOSURE is not supported by this sensor" << std::endl;
    }
    if (sensor.supports(RS2_OPTION_GAIN)) {
        try {
            sensor.set_option(RS2_OPTION_GAIN, 128);
        } catch (const rs2::error& e) {
            std::cerr << "Failed to set option " << RS2_OPTION_GAIN << ". (" << e.what() << ")" << std::endl;
        }
    } else {
        std::cerr << "RS2_OPTION_GAIN is not supported by this sensor" << std::endl;
    }
    if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
        try {
            sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 1);
        } catch (const rs2::error& e) {
            std::cerr << "Failed to set option " << RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE << ". (" << e.what() << ")" << std::endl;
        }
    } else {
        std::cerr << "RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE is not supported by this sensor" << std::endl;
    }
#if (0)
    if (sensor.supports(RS2_OPTION_WHITE_BALANCE)) {
        try {
            sensor.set_option(RS2_OPTION_WHITE_BALANCE, 2100.0);
        } catch (const rs2::error& e) {
            std::cerr << "Failed to set option " << RS2_OPTION_WHITE_BALANCE << ". (" << e.what() << ")" << std::endl;
        }
    } else {
        std::cerr << "RS2_OPTION_WHITE_BALANCE is not supported by this sensor" << std::endl;
    }
#endif
    /*
        Original RealSense example used 640x480 depth and 1920x1080 color configuration.
        Use the available higher depth resolutions for this example.
    */
    rs2::config configuration;
    configuration.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
    configuration.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 30);

    rs2::pipeline pipe;                            // Create a pipeline to easily configure and start the camera.
    rs2::pipeline_profile profile = pipe.start(configuration);  // Start the first connected device with it's default stream.

    std::cout << "\nAvailable (type, format) streams in the profile:" << std::endl;
    for (rs2::stream_profile sp : profile.get_streams()) {
        int i = 0;
        std::cout << "  " << i++ << " " << rs2_stream_to_string(sp.stream_type()) << ", " << rs2_format_to_string(sp.format())
                  << std::endl;
    }

    /*
        Get the size (width,height) of frames for visualization. 
        Could get this from the profile and avoid the wait_for_frames.
    */
    rs2::frameset frametemp = pipe.wait_for_frames();
    auto depth = frametemp.get_depth_frame();
    auto color = frametemp.get_color_frame();

    auto depth_image_ptr = std::make_shared<geometry::Image>();
    depth_image_ptr->Prepare(depth.get_width(), depth.get_height(), 1, 2);
    auto color_image_ptr = std::make_shared<geometry::Image>();
    color_image_ptr->Prepare(color.get_width(), color.get_height(), 3, 1);

    /*
        Open up windows for depth and color visualization.
     */
    visualization::Visualizer depth_vis, color_vis;
    if (depth_vis.CreateVisualizerWindow("Depth", depth.get_width(), depth.get_height(), 15, 50) == false ||
        depth_vis.AddGeometry(depth_image_ptr) == false ||
        color_vis.CreateVisualizerWindow("Color", color.get_width(), color.get_height(), 675, 50) == false ||
        color_vis.AddGeometry(color_image_ptr) == false) {
        return 0;
    }

    /*
        Process frames in this loop until a window is closed.
    */
    while (depth_vis.PollEvents() && color_vis.PollEvents()) {
        rs2::frameset frameset = pipe.wait_for_frames();

        depth = frameset.get_depth_frame();
        color = frameset.get_color_frame();

        (void *)memcpy(depth_image_ptr->data_.data(), depth.get_data(), (size_t)depth.get_data_size());
        (void *)memcpy(color_image_ptr->data_.data(), color.get_data(), (size_t)color.get_data_size());

        (void)depth_vis.UpdateGeometry();
        color_vis.UpdateGeometry();

        std::cout << "white balance " << sensor.get_option(RS2_OPTION_WHITE_BALANCE) << std::endl; // Auto white balance doesn't appear to be enabled?

    }
    return EXIT_SUCCESS;
} catch (const rs2::error& e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what()
              << std::endl;
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
