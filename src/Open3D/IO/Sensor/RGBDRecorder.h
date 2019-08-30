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

#pragma once

#include "Open3D/IO/Sensor/RGBDSensorConfig.h"

namespace open3d {
namespace io {

class RGBDRecorder {
public:
    RGBDRecorder() {}
    virtual ~RGBDRecorder() {}

    /// Init recorder, connect to sensor
    virtual bool InitSensor() = 0;

    /// Create recording file
    virtual bool OpenRecord(const std::string &filename) = 0;

    /// Record one frame, return an RGBDImage. If \param write is true, the
    /// RGBDImage frame will be written to file.
    /// If \param enable_align_depth_to_color is true, the depth image will be
    /// warped to align with the color image; otherwise the raw depth image
    /// output will be saved. Setting \param enable_align_depth_to_color to
    /// false is useful when recording at high resolution with high frame rates.
    /// In this case, the depth image must be warped to align with the color
    /// image with when reading from the recorded file.
    virtual std::shared_ptr<geometry::RGBDImage> RecordFrame(
            bool write, bool enable_align_depth_to_color) = 0;

    /// Flush data to recording file and disconnect from sensor
    virtual bool CloseRecord() = 0;
};

}  // namespace io
}  // namespace open3d
