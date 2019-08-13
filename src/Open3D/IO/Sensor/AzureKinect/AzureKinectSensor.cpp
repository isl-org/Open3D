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

#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensor.h"
#include "Open3D/Geometry/RGBDImage.h"

#include <k4a/k4a.h>
#include <k4arecord/record.h>
#include <turbojpeg.h>
#include <memory>

// call k4a_device_close on every failed CHECK
#define CHECK(x, device)                                                     \
    {                                                                        \
        auto retval = (x);                                                   \
        if (retval) {                                                        \
            open3d::utility::LogError("Runtime error: {} returned {}\n", #x, \
                                      retval);                               \
                                                                             \
            k4a_device_close(device);                                        \
            return 1;                                                        \
        }                                                                    \
    }

namespace open3d {
namespace io {

AzureKinectSensor::AzureKinectSensor(
        const AzureKinectSensorConfig &sensor_config)
    : RGBDSensor(), sensor_config_(sensor_config) {
    sensor_native_config_ = sensor_config.ConvertToNativeConfig();
}

AzureKinectSensor::~AzureKinectSensor() {
    k4a_device_stop_cameras(device_);
    k4a_device_close(device_);
}

int AzureKinectSensor::Connect(size_t sensor_index) {
    // check mode
    int camera_fps;
    switch (sensor_native_config_.camera_fps) {
        case K4A_FRAMES_PER_SECOND_5:
            camera_fps = 5;
            break;
        case K4A_FRAMES_PER_SECOND_15:
            camera_fps = 15;
            break;
        case K4A_FRAMES_PER_SECOND_30:
            camera_fps = 30;
            break;
        default:
            camera_fps = 30;
            break;
    }
    timeout_ = int(1000.0f / camera_fps);

    if ((sensor_native_config_.color_resolution == K4A_COLOR_RESOLUTION_OFF &&
         sensor_native_config_.depth_mode == K4A_DEPTH_MODE_OFF)) {
        utility::LogError(
                "Config error: either the color or depth modes must be "
                "enabled to record.\n");
        return 1;
    }

    // Check available devices
    const uint32_t installed_devices = k4a_device_get_installed_count();
    if (sensor_index >= installed_devices) {
        utility::LogError("Device not found.\n");
        return 1;
    }

    // Open device
    if (K4A_FAILED(k4a_device_open(sensor_index, &device_))) {
        utility::LogError("Runtime error: k4a_device_open() failed\n");
        return 1;
    }

    // Get and print device info
    char serial_number_buffer[256];
    size_t serial_number_buffer_size = sizeof(serial_number_buffer);
    CHECK(k4a_device_get_serialnum(device_, serial_number_buffer,
                                   &serial_number_buffer_size),
          device_);
    k4a_hardware_version_t version_info;
    CHECK(k4a_device_get_version(device_, &version_info), device_);
    utility::LogInfo("Device serial number: {}\n", serial_number_buffer);
    utility::LogInfo("Device version: {};\n",
                     (version_info.firmware_build == K4A_FIRMWARE_BUILD_RELEASE
                              ? "Rel"
                              : "Dbg"));
    utility::LogInfo("C: {}.{}.{};\n", version_info.rgb.major,
                     version_info.rgb.minor, version_info.rgb.iteration);
    utility::LogInfo("D: {}.{}.{}[{}.{}]\n", version_info.depth.major,
                     version_info.depth.minor, version_info.depth.iteration,
                     version_info.depth_sensor.major,
                     version_info.depth_sensor.minor);
    utility::LogInfo("A: {}.{}.{};\n", version_info.audio.major,
                     version_info.audio.minor, version_info.audio.iteration);

    CHECK(k4a_device_start_cameras(device_, &sensor_native_config_), device_);

    // Set color control, assume absoluteExposureValue == 0
    if (K4A_FAILED(k4a_device_set_color_control(
                device_, K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                K4A_COLOR_CONTROL_MODE_AUTO, 0))) {
        utility::LogError(
                "Runtime error: k4a_device_set_color_control() failed\n");
    }

    // Set calibration
    k4a_calibration_t calibration;
    k4a_device_get_calibration(device_, sensor_native_config_.depth_mode,
                               sensor_native_config_.color_resolution,
                               &calibration);
    transform_depth_to_color_ = k4a_transformation_create(&calibration);

    return 0;
}

std::shared_ptr<geometry::RGBDImage> AzureKinectSensor::CaptureFrame() const {
    k4a_capture_t capture;
    auto result = k4a_device_get_capture(device_, &capture, timeout_);
    if (result == K4A_WAIT_RESULT_TIMEOUT) {
        return nullptr;
    } else if (result != K4A_WAIT_RESULT_SUCCEEDED) {
        utility::LogError(
                "Runtime error: k4a_device_get_capture() returned %d\n",
                result);
        return nullptr;
    }

    /* this is a static ptr and will be updated internally */
    auto im_rgbd = DecompressCapture(capture, transform_depth_to_color_);
    k4a_capture_release(capture);
    return im_rgbd;
}

void AzureKinectSensor::ConvertBGRAToRGB(geometry::Image &rgba,
                                         geometry::Image &rgb) {
    assert(rgba.bytes_per_channel_ == 1 && rgba.num_of_channels_ == 4);
    assert(rgb.bytes_per_channel_ == 1 && rgb.num_of_channels_ == 3);
    assert(rgba.width_ == rgb.width_ && rgba.height_ == rgb.height_);

    int N = rgba.width_ * rgba.height_;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int uv = 0; uv < N; ++uv) {
        int v = uv / rgba.width_;
        int u = uv % rgba.width_;
        for (int c = 0; c < 3; ++c) {
            *rgb.PointerAt<uint8_t>(u, v, c) =
                    *rgba.PointerAt<uint8_t>(u, v, 2 - c);
        }
    }
}

std::shared_ptr<geometry::RGBDImage> AzureKinectSensor::DecompressCapture(
        k4a_capture_t capture, k4a_transformation_t transformation) {
    static std::shared_ptr<geometry::Image> color_buffer = nullptr;
    static std::shared_ptr<geometry::RGBDImage> rgbd_buffer = nullptr;

    if (color_buffer == nullptr) {
        color_buffer = std::make_shared<geometry::Image>();
    }
    if (rgbd_buffer == nullptr) {
        rgbd_buffer = std::make_shared<geometry::RGBDImage>();
    }

    k4a_image_t k4a_color = k4a_capture_get_color_image(capture);
    k4a_image_t k4a_depth = k4a_capture_get_depth_image(capture);
    if (k4a_color == nullptr || k4a_depth == nullptr) {
        utility::LogDebug("Skipping empty captures.\n");
        return nullptr;
    }

    /* Process color */
    if (K4A_IMAGE_FORMAT_COLOR_MJPG != k4a_image_get_format(k4a_color)) {
        utility::LogError(
                "Unexpected image format. The stream may have "
                "corrupted.\n");
        return nullptr;
    }

    int width = k4a_image_get_width_pixels(k4a_color);
    int height = k4a_image_get_height_pixels(k4a_color);

    /* resize */
    rgbd_buffer->color_.Prepare(width, height, 3, sizeof(uint8_t));
    color_buffer->Prepare(width, height, 4, sizeof(uint8_t));

    tjhandle tjHandle;
    tjHandle = tjInitDecompress();
    if (0 !=
        tjDecompress2(tjHandle, k4a_image_get_buffer(k4a_color),
                      static_cast<unsigned long>(k4a_image_get_size(k4a_color)),
                      color_buffer->data_.data(), width, 0 /* pitch */, height,
                      TJPF_BGRA, TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE)) {
        utility::LogError("Failed to decompress color image.\n");
        return nullptr;
    }
    tjDestroy(tjHandle);
    ConvertBGRAToRGB(*color_buffer, rgbd_buffer->color_);

    /* transform depth to color plane */
    k4a_image_t k4a_transformed_depth = nullptr;
    if (transformation) {
        rgbd_buffer->depth_.Prepare(width, height, 1, sizeof(uint16_t));
        k4a_image_create_from_buffer(K4A_IMAGE_FORMAT_DEPTH16, width, height,
                                     width * sizeof(uint16_t),
                                     rgbd_buffer->depth_.data_.data(),
                                     width * height * sizeof(uint16_t), NULL,
                                     NULL, &k4a_transformed_depth);
        if (K4A_RESULT_SUCCEEDED !=
            k4a_transformation_depth_image_to_color_camera(
                    transformation, k4a_depth, k4a_transformed_depth)) {
            utility::LogError(
                    "Failed to transform depth frame to color frame.\n");
            return nullptr;
        }
    } else {
        rgbd_buffer->depth_.Prepare(k4a_image_get_width_pixels(k4a_depth),
                                    k4a_image_get_height_pixels(k4a_depth), 1,
                                    sizeof(uint16_t));
        memcpy(rgbd_buffer->depth_.data_.data(),
               k4a_image_get_buffer(k4a_depth), k4a_image_get_size(k4a_depth));
    }

    /* process depth */
    k4a_image_release(k4a_color);
    k4a_image_release(k4a_depth);
    if (transformation) {
        k4a_image_release(k4a_transformed_depth);
    }

    return rgbd_buffer;
}

}  // namespace io
}  // namespace open3d
