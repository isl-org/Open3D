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

#pragma once

#include <atomic>
#include <condition_variable>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "open3d/io/sensor/RGBDSensorConfig.h"
#include "open3d/t/io/sensor/RGBDVideoReader.h"
#include "open3d/utility/IJsonConvertible.h"

// Forward declarations for librealsense classes
namespace rs2 {
class pipeline;
}  // namespace rs2

namespace open3d {
namespace t {
namespace io {

///  \class RSBagReader
///
/// RealSense Bag file reader.
///
/// Only the first color and depth streams from the bag file will be read.
///  - The streams must have the same frame rate.
///  - The color stream must have RGB 8 bit (RGB8/BGR8) pixel format
///  - The depth stream must have 16 bit unsigned int (Z16) pixel format
/// The output is synchronized color and depth frame pairs with the depth frame
/// aligned to the color frame. Unsynchronized frames will be dropped. With
/// alignment, the depth and color frames have the same viewpoint and
/// resolution. See
/// https://intelrealsense.github.io/librealsense/doxygen/rs__sensor_8h.html#ae04b7887ce35d16dbd9d2d295d23aac7
/// for format documentation
///
/// Note: A few frames may be dropped if user code takes a long time (>10 frame
/// intervals) to process a frame.
///
class RSBagReader : public RGBDVideoReader {
public:
    static const size_t DEFAULT_BUFFER_SIZE = 32;

    /// Constructor
    ///
    /// \param buffer_size (optional) Max number of frames to store in the frame
    /// buffer
    explicit RSBagReader(size_t buffer_size = DEFAULT_BUFFER_SIZE);

    RSBagReader(const RSBagReader &) = delete;
    RSBagReader &operator=(const RSBagReader &) = delete;
    virtual ~RSBagReader();

    /// Check If the RSBag file is opened.
    virtual bool IsOpened() const override { return is_opened_; }

    /// Check if the RSBag file is all read.
    virtual bool IsEOF() const override;

    /// Open an RGBD Video playback.
    ///
    /// \param filename Path to the RSBag file.
    virtual bool Open(const std::string &filename) override;

    /// Close the opened RSBag playback.
    virtual void Close() override;

    /// Get (read-only) metadata of the playback.
    virtual const RGBDVideoMetadata &GetMetadata() const override {
        return metadata_;
    }

    /// Get reference to the metadata of the RGBD video playback.
    virtual RGBDVideoMetadata &GetMetadata() override { return metadata_; }

    /// Seek to the timestamp (in us).
    ///
    /// \param timestamp Time in us to seek to.
    virtual bool SeekTimestamp(uint64_t timestamp) override;

    /// Get current timestamp (in us).
    virtual uint64_t GetTimestamp() const override;

    /// Copy next frame from the bag file and return the RGBDImage object.
    virtual t::geometry::RGBDImage NextFrame() override;

    /// Return filename being read
    virtual std::string GetFilename() const override { return filename_; };

    using RGBDVideoReader::SaveFrames;
    using RGBDVideoReader::ToString;

private:
    std::string filename_;
    RGBDVideoMetadata metadata_;

    std::atomic<bool> is_eof_{false};     // Write by frame_reader_thread.
    std::atomic<bool> is_opened_{false};  // Read by frame_reader_thread.
    /// A frame buffer and a separate frame reader thread are used to prevent
    /// frame drops in non real time applications, when the frames
    /// are processed by user code for an arbitrarily long time. The
    /// librealsense2 API for this use case rs2::playback::set_real_time(false)
    /// results in a deadlock after 4 frames on macOS and Linux with SDK
    /// v2.40.0. The recommended workaround with rs2::playback::pause() and
    /// rs2::playback::resume() after reading each frame results in memory
    /// corruption in macOS (not in Linux).
    /// https://github.com/IntelRealSense/librealsense/issues/7547#issuecomment-706984376
    std::vector<t::geometry::RGBDImage> frame_buffer_;
    std::vector<uint64_t>
            frame_position_us_;  ///< Buffer for frame position in us.
    static const size_t BUFFER_REFILL_FACTOR =
            4;  ///< Refill frame buffer when it is only (eg 1/4th ) full.
    std::atomic<uint64_t> head_fid_{
            0};  ///< Next write position by frame_reader_thread.
    std::atomic<uint64_t> tail_fid_{0};  ///< Next unread frame position.
    std::atomic<uint64_t> seek_to_{UINT64_MAX};
    std::condition_variable need_frames_;
    /// This workaround implements a single producer single consumer frame queue
    /// with a circular buffer. The producer thread (frame_reader_thread) keeps
    /// the buffer full. The main thread reads from the current position in the
    /// buffer and signals the producer thread with a condition variable for
    /// more frames if less than a quarter of the frames remain.
    void fill_frame_buffer();
    std::thread frame_reader_thread_;

    std::unique_ptr<rs2::pipeline> pipe_;

    Json::Value GetMetadataJson();
    std::string GetTagInMetadata(const std::string &tag_name);
};
}  // namespace io
}  // namespace t
}  // namespace open3d
