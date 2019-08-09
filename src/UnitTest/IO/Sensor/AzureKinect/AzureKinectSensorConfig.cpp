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

#include "TestUtility/UnitTest.h"

#include <json/json.h>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <string>
#include <unordered_map>

#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensorConfig.h"

using namespace open3d;

TEST(AzureKinectSensorConfig, DefaultConstructor) {
    std::unordered_map<std::string, std::string> ref_config;
    ref_config["color_format"] = "K4A_IMAGE_FORMAT_COLOR_MJPG";
    ref_config["color_resolution"] = "K4A_COLOR_RESOLUTION_1080P";
    ref_config["depth_mode"] = "K4A_DEPTH_MODE_WFOV_2X2BINNED";
    ref_config["camera_fps"] = "K4A_FRAMES_PER_SECOND_30";
    ref_config["wired_sync_mode"] = "K4A_WIRED_SYNC_MODE_STANDALONE";
    ref_config["depth_delay_off_color_usec"] = "0";
    ref_config["subordinate_delay_off_master_usec"] = "0";
    ref_config["disable_streaming_indicator"] = "false";

    io::AzureKinectSensorConfig kinect_config;
    EXPECT_TRUE(kinect_config.config_ == ref_config);
}

TEST(AzureKinectSensorConfig, CustomConstructor) {
    io::AzureKinectSensorConfig default_config;
    EXPECT_EQ(default_config.config_["color_format"],
              "K4A_IMAGE_FORMAT_COLOR_MJPG");

    std::unordered_map<std::string, std::string> custom_config_map;
    custom_config_map["color_format"] = "K4A_IMAGE_FORMAT_COLOR_NV12";
    io::AzureKinectSensorConfig custom_config(custom_config_map);
    EXPECT_EQ(custom_config.config_["color_format"],
              "K4A_IMAGE_FORMAT_COLOR_NV12");
}

TEST(AzureKinectSensorConfig, ConvertFromNativeConfig) {
    k4a_device_configuration_t k4a_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    k4a_config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    k4a_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
    k4a_config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    k4a_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    k4a_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    k4a_config.depth_delay_off_color_usec = 0;
    k4a_config.subordinate_delay_off_master_usec = 0;
    k4a_config.disable_streaming_indicator = false;

    io::AzureKinectSensorConfig kinect_config;
    kinect_config.ConvertFromNativeConfig(k4a_config);
}

TEST(AzureKinectSensorConfig, ConvertToNativeConfig) {}
