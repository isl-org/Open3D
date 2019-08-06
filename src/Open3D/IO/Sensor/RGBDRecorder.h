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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef RECORDER_H
#define RECORDER_H

#include <k4a/k4a.h>
#include <atomic>
#include <memory>
#include "CmdParser.h"

namespace open3d {

namespace geometry {
class RGBDImage;
class Image;
}  // namespace geometry

namespace io {

extern std::atomic_bool exiting;

void HstackRGBDepth(const std::shared_ptr<geometry::RGBDImage>& im_rgbd,
                    geometry::Image& im_rgb_depth_hstack);

int Record(uint8_t device_index,
           char* recording_filename,
           int recording_length,
           k4a_device_configuration_t* device_config,
           bool record_imu,
           int32_t absoluteExposureValue);
}  // namespace io
}  // namespace open3d

#endif /* RECORDER_H */
