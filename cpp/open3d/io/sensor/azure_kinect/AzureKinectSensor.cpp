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

#include "open3d/io/sensor/azure_kinect/AzureKinectSensor.h"

#include <k4a/k4a.h>
#include <k4arecord/record.h>
#include <turbojpeg.h>

#include <memory>

#include "open3d/geometry/RGBDImage.h"
#include "open3d/io/sensor/azure_kinect/K4aPlugin.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace io {

AzureKinectSensor::AzureKinectSensor(
        const AzureKinectSensorConfig &sensor_config)
    : RGBDSensor(), sensor_config_(sensor_config) {}

AzureKinectSensor::~AzureKinectSensor() { Disconnect(); }

bool AzureKinectSensor::Connect(size_t sensor_index) {
    utility::LogInfo("AzureKinectSensor::Connect");
    utility::LogInfo("sensor_index {}", sensor_index);
    auto device_config = sensor_config_.ConvertToNativeConfig();

    // check mode
    int camera_fps;
    switch (device_config.camera_fps) {
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

    if (device_config.color_resolution == K4A_COLOR_RESOLUTION_OFF &&
        device_config.depth_mode == K4A_DEPTH_MODE_OFF) {
        utility::LogWarning(
                "Config error: either the color or depth modes must be "
                "enabled to record.");
        return false;
    }

    // check available devices
    const uint32_t installed_devices =
            k4a_plugin::k4a_device_get_installed_count();
    if (sensor_index >= installed_devices) {
        utility::LogWarning("Device not found.");
        return false;
    }

    // open device
    if (K4A_FAILED(k4a_plugin::k4a_device_open(
                static_cast<uint32_t>(sensor_index), &device_))) {
        utility::LogWarning(
                "Runtime error: k4a_plugin::k4a_device_open() failed");
        return false;
    }

    // get and print device info
    PrintFirmware(device_);

    if (K4A_FAILED(k4a_plugin::k4a_device_start_cameras(device_,
                                                        &device_config))) {
        utility::LogWarning(
                "Runtime error: k4a_plugin::k4a_device_start_cameras() "
                "failed");
        k4a_plugin::k4a_device_close(device_);
        return false;
    }

    // set color control, assume absoluteExposureValue == 0
    if (K4A_FAILED(k4a_plugin::k4a_device_set_color_control(
                device_, K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                K4A_COLOR_CONTROL_MODE_AUTO, 0))) {
        utility::LogWarning(
                "Runtime error: k4a_plugin::k4a_device_set_color_control() "
                "failed");
        k4a_plugin::k4a_device_close(device_);
        return false;
    }

    // set calibration
    k4a_calibration_t calibration;
    k4a_plugin::k4a_device_get_calibration(device_, device_config.depth_mode,
                                           device_config.color_resolution,
                                           &calibration);
    transform_depth_to_color_ =
            k4a_plugin::k4a_transformation_create(&calibration);

    return true;
}

void AzureKinectSensor::Disconnect() {
    k4a_plugin::k4a_device_stop_cameras(device_);
    k4a_plugin::k4a_device_close(device_);
}

k4a_capture_t AzureKinectSensor::CaptureRawFrame() const {
    k4a_capture_t capture;
    auto result =
            k4a_plugin::k4a_device_get_capture(device_, &capture, timeout_);
    if (result == K4A_WAIT_RESULT_TIMEOUT) {
        return nullptr;
    } else if (result != K4A_WAIT_RESULT_SUCCEEDED) {
        utility::LogWarning(
                "Runtime error: k4a_plugin::k4a_device_get_capture() returned "
                "{}",
                result);
        return nullptr;
    }

    return capture;
}

std::shared_ptr<geometry::RGBDImage> AzureKinectSensor::CaptureFrame(
        bool enable_align_depth_to_color) const {
    k4a_capture_t capture = CaptureRawFrame();
    if (!capture) return nullptr;

    auto im_rgbd = DecompressCapture(
            capture,
            enable_align_depth_to_color ? transform_depth_to_color_ : nullptr);
    k4a_plugin::k4a_capture_release(capture);
    return im_rgbd;
}

void ConvertBGRAToRGB(geometry::Image &bgra, geometry::Image &rgb) {
    if (bgra.bytes_per_channel_ != 1) {
        utility::LogError("BGRA input image must have 1 byte per channel.");
    }
    if (rgb.bytes_per_channel_ != 1) {
        utility::LogError("RGB output image must have 1 byte per channel.");
    }
    if (bgra.num_of_channels_ != 4) {
        utility::LogError("BGRA input image must have 4 channels.");
    }
    if (rgb.num_of_channels_ != 3) {
        utility::LogError("RGB output image must have 3 channels.");
    }
    if (bgra.width_ != rgb.width_ || bgra.height_ != rgb.height_) {
        utility::LogError(
                "BGRA input image and RGB output image have different "
                "dimensions.");
    }

#ifdef _WIN32
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
#else
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (int v = 0; v < bgra.height_; ++v) {
        for (int u = 0; u < bgra.width_; ++u) {
            for (int c = 0; c < 3; ++c) {
                *rgb.PointerAt<uint8_t>(u, v, c) =
                        *bgra.PointerAt<uint8_t>(u, v, 2 - c);
            }
        }
    }
}

bool AzureKinectSensor::PrintFirmware(k4a_device_t device) {
    char serial_number_buffer[256];
    size_t serial_number_buffer_size = sizeof(serial_number_buffer);
    if (K4A_BUFFER_RESULT_SUCCEEDED !=
        k4a_plugin::k4a_device_get_serialnum(device, serial_number_buffer,
                                             &serial_number_buffer_size)) {
        utility::LogWarning(
                "Runtime error: k4a_plugin::k4a_device_get_serialnum() "
                "failed");
        return false;
    }

    k4a_hardware_version_t version_info;
    if (K4A_FAILED(k4a_plugin::k4a_device_get_version(device, &version_info))) {
        utility::LogWarning(
                "Runtime error: k4a_plugin::k4a_device_get_version() failed");
        return false;
    }

    utility::LogInfo("Serial number: {}", serial_number_buffer);
    utility::LogInfo("Firmware build: {}",
                     (version_info.firmware_build == K4A_FIRMWARE_BUILD_RELEASE
                              ? "Rel"
                              : "Dbg"));
    utility::LogInfo("> Color: {}.{}.{}", version_info.rgb.major,
                     version_info.rgb.minor, version_info.rgb.iteration);
    utility::LogInfo("> Depth: {}.{}.{}[{}.{}]", version_info.depth.major,
                     version_info.depth.minor, version_info.depth.iteration,
                     version_info.depth_sensor.major,
                     version_info.depth_sensor.minor);
    return true;
}

bool AzureKinectSensor::ListDevices() {
    uint32_t device_count = k4a_plugin::k4a_device_get_installed_count();
    if (device_count > 0) {
        for (uint8_t i = 0; i < device_count; i++) {
            utility::LogInfo("Device index {}", i);
            k4a_device_t device;
            if (K4A_SUCCEEDED(k4a_plugin::k4a_device_open(i, &device))) {
                AzureKinectSensor::PrintFirmware(device);
                k4a_plugin::k4a_device_close(device);
            } else {
                utility::LogWarning("Device Open Failed");
                return false;
            }
        }
    } else {
        utility::LogError("No devices connected.");
    }

    return true;
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

    k4a_image_t k4a_color = k4a_plugin::k4a_capture_get_color_image(capture);
    k4a_image_t k4a_depth = k4a_plugin::k4a_capture_get_depth_image(capture);
    if (k4a_color == nullptr || k4a_depth == nullptr) {
        utility::LogDebug("Skipping empty captures.");
        return nullptr;
    }

    /* Process color */
    if (K4A_IMAGE_FORMAT_COLOR_MJPG !=
        k4a_plugin::k4a_image_get_format(k4a_color)) {
        utility::LogWarning(
                "Unexpected image format. The stream may have been corrupted.");
        return nullptr;
    }

    int width = k4a_plugin::k4a_image_get_width_pixels(k4a_color);
    int height = k4a_plugin::k4a_image_get_height_pixels(k4a_color);

    /* resize */
    rgbd_buffer->color_.Prepare(width, height, 3, sizeof(uint8_t));
    color_buffer->Prepare(width, height, 4, sizeof(uint8_t));

    tjhandle tjHandle;
    tjHandle = tjInitDecompress();
    if (0 !=
        tjDecompress2(tjHandle, k4a_plugin::k4a_image_get_buffer(k4a_color),
                      static_cast<unsigned long>(
                              k4a_plugin::k4a_image_get_size(k4a_color)),
                      color_buffer->data_.data(), width, 0 /* pitch */, height,
                      TJPF_BGRA, TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE)) {
        utility::LogWarning("Failed to decompress color image.");
        return nullptr;
    }
    tjDestroy(tjHandle);
    ConvertBGRAToRGB(*color_buffer, rgbd_buffer->color_);

    /* transform depth to color plane */
    k4a_image_t k4a_transformed_depth = nullptr;
    if (transformation) {
        rgbd_buffer->depth_.Prepare(width, height, 1, sizeof(uint16_t));
        k4a_plugin::k4a_image_create_from_buffer(
                K4A_IMAGE_FORMAT_DEPTH16, width, height,
                width * sizeof(uint16_t), rgbd_buffer->depth_.data_.data(),
                width * height * sizeof(uint16_t), NULL, NULL,
                &k4a_transformed_depth);
        if (K4A_RESULT_SUCCEEDED !=
            k4a_plugin::k4a_transformation_depth_image_to_color_camera(
                    transformation, k4a_depth, k4a_transformed_depth)) {
            utility::LogWarning(
                    "Failed to transform depth frame to color frame.");
            return nullptr;
        }
    } else {
        rgbd_buffer->depth_.Prepare(
                k4a_plugin::k4a_image_get_width_pixels(k4a_depth),
                k4a_plugin::k4a_image_get_height_pixels(k4a_depth), 1,
                sizeof(uint16_t));
        memcpy(rgbd_buffer->depth_.data_.data(),
               k4a_plugin::k4a_image_get_buffer(k4a_depth),
               k4a_plugin::k4a_image_get_size(k4a_depth));
    }

    /* process depth */
    k4a_plugin::k4a_image_release(k4a_color);
    k4a_plugin::k4a_image_release(k4a_depth);
    if (transformation) {
        k4a_plugin::k4a_image_release(k4a_transformed_depth);
    }

    return rgbd_buffer;
}

}  // namespace io
}  // namespace open3d
