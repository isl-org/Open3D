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

#include "open3d/t/io/sensor/realsense/RSBagReader.h"

#include <json/json.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <tuple>

#include "open3d/t/io/sensor/realsense/RealSensePrivate.h"
#include "open3d/t/io/sensor/realsense/RealSenseSensorConfig.h"

namespace open3d {
namespace t {
namespace io {

RSBagReader::RSBagReader(size_t buffer_size)
    : frame_buffer_(buffer_size),
      frame_position_us_(buffer_size),
      pipe_(nullptr) {}

RSBagReader::~RSBagReader() {
    if (IsOpened()) Close();
}

bool RSBagReader::Open(const std::string &filename) {
    if (IsOpened()) {
        Close();
    }
    try {
        rs2::config cfg;
        cfg.enable_device_from_file(filename, false);  // Do not repeat playback
        cfg.disable_all_streams();
        cfg.enable_stream(RS2_STREAM_DEPTH);
        cfg.enable_stream(RS2_STREAM_COLOR);
        pipe_.reset(new rs2::pipeline);
        auto profile = pipe_->start(cfg);  // Open file in read mode
        // do not drop frames: Causes deadlock after 4 frames on macOS/Linux
        // https://github.com/IntelRealSense/librealsense/issues/7547#issuecomment-706984376
        /* profile.get_device().as<rs2::playback>().set_real_time(false); */
        metadata_.ConvertFromJsonValue(
                RealSenseSensorConfig::GetMetadataJson(profile));
        RealSenseSensorConfig::GetPixelDtypes(profile, metadata_);
        if (seek_to_ == UINT64_MAX) {
            utility::LogInfo("File {} opened", filename);
        }
    } catch (const rs2::error &) {
        utility::LogWarning("Unable to open file {}", filename);
        return false;
    }
    filename_ = filename;
    is_eof_ = false;
    is_opened_ = true;
    // Launch thread to keep frame_buffer full
    frame_reader_thread_ = std::thread{&RSBagReader::fill_frame_buffer, this};
    return true;
}

void RSBagReader::Close() {
    is_opened_ = false;
    need_frames_.notify_one();
    frame_reader_thread_.join();
    pipe_->stop();
}

void RSBagReader::fill_frame_buffer() try {
    std::mutex frame_buffer_mutex;
    std::unique_lock<std::mutex> lock(frame_buffer_mutex);
    const unsigned int RS2_PLAYBACK_TIMEOUT_MS =
            static_cast<unsigned int>(10 * 1000.0 / metadata_.fps_);
    rs2::frameset frames;
    rs2::playback rs_device =
            pipe_->get_active_profile().get_device().as<rs2::playback>();
    rs2::align align_to_color(rs2_stream::RS2_STREAM_COLOR);
    head_fid_ = 0;
    tail_fid_ = 0;
    uint64_t next_dev_color_fid = 0;
    uint64_t dev_color_fid = 0;

    while (is_opened_) {
        rs_device.resume();
        utility::LogDebug(
                "frame_reader_thread_ start reading tail_fid_={}, "
                "head_fid_={}.",
                tail_fid_, head_fid_);
        while (!is_eof_ && head_fid_ < tail_fid_ + frame_buffer_.size()) {
            if (seek_to_ < UINT64_MAX) {
                utility::LogDebug("frame_reader_thread_ seek to {}us",
                                  seek_to_);
                rs_device.seek(std::chrono::microseconds(seek_to_));
                tail_fid_.store(
                        head_fid_.load());  // atomic: Invalidate buffer.
                next_dev_color_fid = dev_color_fid = 0;
                seek_to_ = UINT64_MAX;
            }
            // Ensure next frameset is not a repeat
            while (next_dev_color_fid == dev_color_fid &&
                   pipe_->try_wait_for_frames(&frames,
                                              RS2_PLAYBACK_TIMEOUT_MS)) {
                next_dev_color_fid =
                        frames.get_color_frame().get_frame_number();
            }
            if (next_dev_color_fid != dev_color_fid) {
                dev_color_fid = next_dev_color_fid;
                auto &current_frame =
                        frame_buffer_[head_fid_ % frame_buffer_.size()];

                frames = align_to_color.process(frames);
                const auto &color_frame = frames.get_color_frame();
                // Copy frame data to Tensors
                current_frame.color_ = core::Tensor(
                        static_cast<const uint8_t *>(color_frame.get_data()),
                        {color_frame.get_height(), color_frame.get_width(),
                         metadata_.color_channels_},
                        metadata_.color_dt_);
                const auto &depth_frame = frames.get_depth_frame();
                current_frame.depth_ = core::Tensor(
                        static_cast<const uint16_t *>(depth_frame.get_data()),
                        {depth_frame.get_height(), depth_frame.get_width()},
                        metadata_.depth_dt_);
                frame_position_us_[head_fid_ % frame_buffer_.size()] =
                        rs_device.get_position() /
                        1000;  // Convert nanoseconds -> microseconds
                ++head_fid_;   // atomic
            } else {
                utility::LogDebug("frame_reader_thread EOF. Join.");
                is_eof_ = true;
                return;
            }
            if (!is_opened_) break;  // exit if Close()
        }
        rs_device.pause();  // Pause playback to prevent frame drops
        utility::LogDebug(
                "frame_reader_thread pause reading tail_fid_={}, head_fid_={}",
                tail_fid_, head_fid_);
        need_frames_.wait(lock, [this] {
            return !is_opened_ ||
                   head_fid_ < tail_fid_ + frame_buffer_.size() /
                                                   BUFFER_REFILL_FACTOR;
        });
    }
} catch (const rs2::error &e) {
    utility::LogError("Realsense error: {}: {}",
                      rs2_exception_type_to_string(e.get_type()), e.what());
} catch (const std::exception &e) {
    utility::LogError("Error in reading RealSense bag file: {}", e.what());
}

bool RSBagReader::IsEOF() const { return is_eof_ && tail_fid_ == head_fid_; }

t::geometry::RGBDImage RSBagReader::NextFrame() {
    if (!IsOpened()) {
        utility::LogError("Null file handler. Please call Open().");
    }
    if (!is_eof_ &&
        head_fid_ < tail_fid_ + frame_buffer_.size() / BUFFER_REFILL_FACTOR)
        need_frames_.notify_one();

    while (!is_eof_ &&
           tail_fid_ ==
                   head_fid_) {  // (rare) spin wait for frame_reader_thread_
        std::this_thread::sleep_for(
                std::chrono::duration<double>(1 / metadata_.fps_));
    }
    if (is_eof_ && tail_fid_ == head_fid_) {  // no more frames
        utility::LogInfo("EOF reached");
        return t::geometry::RGBDImage();
    } else {
        return frame_buffer_[(tail_fid_++) %  // atomic
                             frame_buffer_.size()];
    }
}

bool RSBagReader::SeekTimestamp(uint64_t timestamp) {
    if (!IsOpened()) {
        utility::LogWarning("Null file handler. Please call Open().");
        return false;
    }
    if (timestamp >= metadata_.stream_length_usec_) {
        utility::LogWarning("Timestamp {} exceeds maximum {} (us).", timestamp,
                            metadata_.stream_length_usec_);
        return false;
    }
    seek_to_ = timestamp;  // atomic
    if (is_eof_) {
        Open(filename_);  // EOF requires restarting pipeline.
    } else {
        need_frames_.notify_one();
    }
    return true;
}

uint64_t RSBagReader::GetTimestamp() const {
    if (!IsOpened()) {
        utility::LogWarning("Null file handler. Please call Open().");
        return UINT64_MAX;
    }
    return tail_fid_ == 0
                   ? 0
                   : frame_position_us_[(tail_fid_ - 1) % frame_buffer_.size()];
}

}  // namespace io
}  // namespace t
}  // namespace open3d
