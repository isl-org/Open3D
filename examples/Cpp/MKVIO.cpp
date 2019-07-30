// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <cstddef>
#include <cstdio>

#include "Open3D/Open3D.h"

#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>
#include <libjpeg-turbo/turbojpeg.h>
#include <iostream>
#include "json/json.h"

namespace open3d {

std::shared_ptr<geometry::Image> ConvertBGRAToRGB(
        std::shared_ptr<geometry::Image> &rgba) {
    assert(rgba->bytes_per_channel_ == 1 && rgba->num_of_channels_ == 4);

    auto rgb = std::make_shared<geometry::Image>();
    rgb->Prepare(rgba->width_, rgba->height_, 3, rgba->bytes_per_channel_);

    for (int u = 0; u < rgba->width_; ++u) {
        for (int v = 0; v < rgba->height_; ++v) {
            for (int c = 0; c < 3; ++c) {
                *rgb->PointerAt<uint8_t>(u, v, c) =
                        *rgba->PointerAt<uint8_t>(u, v, c);
            }
        }
    }

    return rgb;
}

class RGBDStreamMetadata : public utility::IJsonConvertible {
public:
    bool ConvertToJsonValue(Json::Value &value) const override {
        intrinsics_.ConvertToJsonValue(value);
        value["serial_number_"] = serial_number_;
        value["stream_length_usec"] = stream_length_usec_;
        return true;
    }
    bool ConvertFromJsonValue(const Json::Value &value) override {
        intrinsics_.ConvertFromJsonValue(value);
        serial_number_ = value["serial_number"].asString();
        stream_length_usec_ = value["stream_length_usec"].asUInt64();
        return true;
    }

public:
    // shared intrinsics betwee RGB & depth.
    // We assume depth image is always warped to the color image system
    camera::PinholeCameraIntrinsic intrinsics_;

    std::string serial_number_ = "";
    uint64_t stream_length_usec_ = 0;
};

class MKVReader {
private:
    k4a_playback_t handle_ = nullptr;
    k4a_transformation_t transformation_ = nullptr;
    RGBDStreamMetadata metadata_;

    std::string GetTagInMetadata(const std::string &tag_name) {
        char res_buffer[256];
        size_t res_size = 256;

        k4a_buffer_result_t result = k4a_playback_get_tag(
                handle_, tag_name.c_str(), res_buffer, &res_size);
        if (K4A_BUFFER_RESULT_SUCCEEDED == result) {
            return res_buffer;
        } else if (K4A_BUFFER_RESULT_TOO_SMALL == result) {
            utility::LogError("{} tag's content is too long.\n",
                              tag_name.c_str());
            return "";
        } else {
            utility::LogError("{} tag does not exist.\n", tag_name.c_str());
            return "";
        }
    }

public:
    bool IsOpened() { return handle_ != nullptr; }

    int Open(const std::string &filename) {
        if (IsOpened()) {
            Close();
        }

        if (K4A_RESULT_SUCCEEDED !=
            k4a_playback_open(filename.c_str(), &handle_)) {
            utility::LogError("Unable to open file {}\n", filename.c_str());
            return -1;
        }

        metadata_.ConvertFromJsonValue(GetMetaData());
        return 0;
    }

    void Close() { k4a_playback_close(handle_); }

    Json::Value GetMetaData() {
        if (!IsOpened()) {
            utility::LogError("Null file handler. Please call Open().\n");
            return Json::Value();
        }

        Json::Value value;
        value["stream_length_usec"] =
                k4a_playback_get_last_timestamp_usec(handle_);
        value["serial_number"] = GetTagInMetadata("K4A_DEVICE_SERIAL_NUMBER");
        value["depth_mode"] = GetTagInMetadata("K4A_DEPTH_MODE");
        value["color_mode"] = GetTagInMetadata("K4A_COLOR_MODE");
        value["ir_mode"] = GetTagInMetadata("K4A_IR_MODE");
        value["imu_mode"] = GetTagInMetadata("K4A_IMU_MODE");

        k4a_calibration_t calibration;
        if (K4A_RESULT_SUCCEEDED !=
            k4a_playback_get_calibration(handle_, &calibration)) {
            utility::LogError("Failed to get calibration\n");
        }

        camera::PinholeCameraIntrinsic pinhole_camera;
        auto color_camera_calibration = calibration.color_camera_calibration;
        auto param = color_camera_calibration.intrinsics.parameters.param;
        pinhole_camera.SetIntrinsics(color_camera_calibration.resolution_width,
                                     color_camera_calibration.resolution_height,
                                     param.fx, param.fy, param.cx, param.cy);
        pinhole_camera.ConvertToJsonValue(value);

        /** For internal usages */
        transformation_ = k4a_transformation_create(&calibration);

        return value;
    }

    int SeekTimestamp(size_t timestamp) {
        if (!IsOpened()) {
            utility::LogError("Null file handler. Please call Open().\n");
            return -1;
        }

        if (timestamp >= metadata_.stream_length_usec_) {
            utility::LogError("Timestamp {} exceeds maximum {} (us).\n",
                    timestamp, metadata_.stream_length_usec_);
            return -1;
        }

        if (K4A_RESULT_SUCCEEDED !=
            k4a_playback_seek_timestamp(handle_, timestamp,
                                        K4A_PLAYBACK_SEEK_BEGIN)) {
            utility::LogError("Unable to go to timestamp {}\n", timestamp);
            return -1;
        }
        return 0;
    }

    std::shared_ptr<geometry::RGBDImage> Next() {
        auto color = std::make_shared<geometry::Image>();
        auto depth = std::make_shared<geometry::Image>();

        if (!IsOpened()) {
            utility::LogError("Null file handler. Please call Open().\n");
            return nullptr;
        }

        k4a_capture_t k4a_capture;
        if (K4A_STREAM_RESULT_SUCCEEDED !=
            k4a_playback_get_next_capture(handle_, &k4a_capture)) {
            utility::LogDebug("Final frame reached\n");
            return nullptr;
        }

        k4a_image_t k4a_color = k4a_capture_get_color_image(k4a_capture);
        k4a_image_t k4a_depth = k4a_capture_get_depth_image(k4a_capture);
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
        int width = metadata_.intrinsics_.width_;
        int height = metadata_.intrinsics_.height_;
        color->Prepare(width, height, 4, sizeof(uint8_t));
        tjhandle tjHandle;
        tjHandle = tjInitDecompress();
        if (0 !=
            tjDecompress2(
                    tjHandle, k4a_image_get_buffer(k4a_color),
                    static_cast<unsigned long>(k4a_image_get_size(k4a_color)),
                    color->data_.data(), width, 0 /* pitch */, height,
                    TJPF_BGRA, TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE)) {
            utility::LogError("Failed to decompress color image.\n");
            return nullptr;
        }
        tjDestroy(tjHandle);
        color = ConvertBGRAToRGB(color);

        /* transform depth to color */
        k4a_image_t k4a_transformed_depth = nullptr;
        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                     width, height,
                                                     width * sizeof(uint16_t),
                                                     &k4a_transformed_depth)) {
            utility::LogError(
                    "Failed to create transformed depth image buffer.\n");
            return nullptr;
        }
        if (K4A_RESULT_SUCCEEDED !=
            k4a_transformation_depth_image_to_color_camera(
                    transformation_, k4a_depth, k4a_transformed_depth)) {
            utility::LogError(
                    "Failed to transform depth frame to color frame.\n");
            return nullptr;
        }

        /* process depth */
        depth->Prepare(width, height, 1, sizeof(uint16_t));
        memcpy(depth->data_.data(), k4a_image_get_buffer(k4a_transformed_depth),
               width * height * 1 * sizeof(uint16_t));

        k4a_image_release(k4a_color);
        k4a_image_release(k4a_depth);
        k4a_image_release(k4a_transformed_depth);
        k4a_capture_release(k4a_capture);

        return geometry::RGBDImage::CreateFromColorAndDepth(*color, *depth);
    }
};
}  // namespace open3d

int main(int argc, char **argv) {
    using namespace open3d;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    MKVReader mkv_reader;
    mkv_reader.Open("/home/dongw1/Workspace/kinect/data/test.mkv");

    auto json = mkv_reader.GetMetaData();
    for (auto iter = json.begin(); iter != json.end(); ++iter) {
        std::cout << iter.key() << " " << json[iter.name()] << "\n";
    }


    std::vector<unsigned long long > timestamps = {15462462346L, 412423, 124200, 0, 12400000};
    for (auto &ts : timestamps) {
        std::cout << "timestamp = " << ts << "ms\n";
        mkv_reader.SeekTimestamp(ts);
        auto rgbd = mkv_reader.Next();
        if (rgbd) {
            auto color = std::make_shared<geometry::Image>(rgbd->color_);
            visualization::DrawGeometries({color});
        } else {
            utility::LogDebug("Null RGBD frame for timestamp {} (us)\n", ts);
        }
    }

    mkv_reader.Close();
}