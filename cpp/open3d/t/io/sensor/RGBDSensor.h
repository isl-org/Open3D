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

namespace open3d {
namespace t {
namespace io {

class RGBDSensor {
public:
    RGBDSensor() {}
    virtual ~RGBDSensor(){};

    /// List all supported devices (currently only RealSense)
    static bool ListDevices();

    virtual bool InitSensor(
            const RGBDConfigConfig &sensor_config = RGBDSensorConfig{},
            size_t sensor_index = 0,
            const std::string &filename = std::string{}) = 0;

    virtual bool StartCapture(bool start_record = false) = 0;

    virtual void PauseRecord() = 0;

    virtual void ResumeRecord() = 0;

    /** Acquire the next synchronized RGBD frameset from the camera.
     *
     * \param align_depth_to_color Enable aligning WFOV depth image to
     * the color image in visualizer.
     * \param wait If true wait for the next frame set, else return immediately
     * with an empty RGBDImage if it is not yet available
     */
    virtual geometry::RGBDImage CaptureFrame(
            bool wait = true, bool align_depth_to_color = true) = 0;

    virtual void StopCapture() = 0;
};

}  // namespace io
}  // namespace t
}  // namespace open3d
