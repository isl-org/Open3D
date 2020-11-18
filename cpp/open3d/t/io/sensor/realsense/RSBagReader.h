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

#include <librealsense2/rs.hpp>
#include <string>
#include <unordered_map>

#include "open3d/io/sensor/RGBDSensorConfig.h"
#include "open3d/t/io/sensor/RGBDVideoReader.h"
#include "open3d/utility/IJsonConvertible.h"

namespace open3d {
namespace t {
namespace io {

/// \class RSBagReader
///
/// RealSense Bag file reader.
class RSBagReader : public RGBDVideoReader {
public:
    /// Names for common RealSense pixel formats. See
    /// https://intelrealsense.github.io/librealsense/doxygen/rs__sensor_8h.html#ae04b7887ce35d16dbd9d2d295d23aac7
    /// for format documentation
public:
    /// \brief Default Constructor.
    RSBagReader();
    virtual ~RSBagReader() {}

    /// Check If the RSBag file is opened.
    virtual bool IsOpened() const override { return is_opened_; }
    /// Check if the RSBag file is all read.
    virtual bool IsEOF() const override { return is_eof_; }

    /// Open an RGBD Video playback.
    ///
    /// \param filename Path to the RSBag file.
    virtual bool Open(const std::string &filename) override;
    /// Close the opened RSBag playback.
    virtual void Close() override;

    /// Get metadata of the playback.
    virtual RGBDVideoMetadata &GetMetadata() override { return metadata_; }
    /// Seek to the timestamp (in us).
    virtual bool SeekTimestamp(size_t timestamp) override;
    /// Copy next frame from the bag file and return the RGBDImage object.
    virtual t::geometry::RGBDImage NextFrame() override;

private:
    RGBDVideoMetadata metadata_;
    bool is_eof_ = false, is_opened_ = false;
    rs2::pipeline pipe_;
    rs2::align align_to_color_;
    rs2::frameset frames_;
    core::Dtype dt_color_, dt_depth_;
    uint8_t channels_color_;

    Json::Value GetMetadataJson();
    std::string GetTagInMetadata(const std::string &tag_name);
};  // namespace io
}  // namespace io
}  // namespace t
}  // namespace open3d
