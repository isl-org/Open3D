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

#include "open3d/io/sensor/RGBDSensorConfig.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/io/sensor/RGBDVideoMetadata.h"

namespace open3d {
namespace t {
namespace io {

class RGBDVideoReader {
public:
    RGBDVideoReader() {}
    virtual ~RGBDVideoReader() {}

    /// Check If the RGBD video file is opened.
    virtual bool IsOpened() const = 0;

    /// Check if the RGBD video file is all read.
    virtual bool IsEOF() const = 0;

    /// Open an RGBD video playback.
    ///
    /// \param filename Path to the RGBD video file.
    virtual bool Open(const std::string &filename) = 0;

    /// Close the opened RGBD video playback.
    virtual void Close() = 0;

    /// Get reference to the metadata of the RGBD video playback.
    virtual RGBDVideoMetadata &GetMetadata() = 0;

    /// Get metadata of the RGBD video playback.
    virtual const RGBDVideoMetadata &GetMetadata() const = 0;

    /// Seek to the timestamp (in us).
    virtual bool SeekTimestamp(uint64_t timestamp) = 0;

    /// Get current timestamp (in us).
    virtual uint64_t GetTimestamp() const = 0;

    /// Get next frame from the RGBD video playback and returns the RGBD object.
    virtual t::geometry::RGBDImage NextFrame() = 0;

    /// Save synchronized and aligned individual frames to subfolders.
    ///
    /// \param frame_path Frames will be stored in stream subfolders 'color' and
    /// 'depth' here. The intrinsic camera calibration for the color stream will
    /// be saved in 'intrinsic.json'.
    //
    /// \param start_time_us (default 0) Start saving frames from this time (us)
    //
    /// \param end_time_us (default video length) Save frames till this time
    /// (us)
    virtual void SaveFrames(const std::string &frame_path,
                            uint64_t start_time_us = 0,
                            uint64_t end_time_us = UINT64_MAX);

    /// Return filename being read.
    virtual std::string GetFilename() const = 0;

    /// Text description.
    virtual std::string ToString() const;

    /// Factory function to create object based on RGBD video file type.
    static std::unique_ptr<RGBDVideoReader> Create(const std::string &filename);
};

}  // namespace io
}  // namespace t
}  // namespace open3d
