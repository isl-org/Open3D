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
class align;
class frameset;
}  // namespace rs2

namespace open3d {
namespace t {
namespace io {

/*! \class RSBagReader
 *
 * RealSense Bag file reader.
 *
 * Only the first color and depth streams from the bag file will be read.
 *  - The streams must have the same frame rate.
 *  - The color stream must have RGB 8 bit (RGB8/BGR8) pixel format
 *  - The depth stream must have 16 bit unsigned int (Z16) pixel format
 * https://intelrealsense.github.io/librealsense/doxygen/rs__sensor_8h.html#ae04b7887ce35d16dbd9d2d295d23aac7
 * for format documentation
 */
class RSBagReader : public RGBDVideoReader {
public:
    static const size_t DEFAULT_BUFFER_SIZE = 64;
    RSBagReader(size_t buffer_size = DEFAULT_BUFFER_SIZE);
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

    /// Get metadata of the playback.
    virtual RGBDVideoMetadata &GetMetadata() override { return metadata_; }
    /// Seek to the timestamp (in us).
    ///
    /// \param timestamp Time in us to seek to
    virtual bool SeekTimestamp(uint64_t timestamp) override;
    /// Get current timestamp (in us).
    virtual uint64_t GetTimestamp() const override;
    /// Copy next frame from the bag file and return the RGBDImage object.
    virtual t::geometry::RGBDImage NextFrame() override;
    using RGBDVideoReader::SaveFrames;

private:
    RGBDVideoMetadata metadata_;
    core::Dtype dt_color_, dt_depth_;
    uint8_t channels_color_;

    bool is_eof_ = false,  ///< Write by frame_reader_thread
            is_opened_ = false;

    std::vector<t::geometry::RGBDImage> frame_buffer_;
    std::vector<uint64_t> frame_position_us_;
    std::atomic<uint64_t> head_fid{0},  ///< Write by frame_reader_thread.
                                        ///< Invalid buffer position.
            tail_fid{0};                ///< Next unread frame position
    std::condition_variable need_frames;
    void fill_frame_buffer(uint64_t start_time_us = 0);
    std::thread frame_reader_thread;

    std::unique_ptr<rs2::pipeline> pipe_;
    std::unique_ptr<rs2::align> align_to_color_;

    Json::Value GetMetadataJson();
    bool Open(const std::string &filename, uint64_t start_time_us);
    std::string GetTagInMetadata(const std::string &tag_name);
};
}  // namespace io
}  // namespace t
}  // namespace open3d

/* frame_acq thread jobs:
 *  - Fill frame_buffer_
 *  - seek
 */
