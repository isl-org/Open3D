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

#include <k4a/k4a.h>
#include <k4arecord/playback.h>

#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/IO/Sensor/MKVMetadata.h"
#include "Open3D/Utility/IJsonConvertible.h"

namespace open3d {
namespace io {

class MKVReader {
public:
    /* Also shared by other RGBDSensor */
    static std::shared_ptr<geometry::RGBDImage> DecompressCapture(
            k4a_capture_t capture, k4a_transformation_t transformation);

    bool IsOpened() { return handle_ != nullptr; }

    int Open(const std::string &filename);
    void Close();

    Json::Value GetMetaData();
    int SeekTimestamp(size_t timestamp);
    std::shared_ptr<geometry::RGBDImage> NextFrame();

private:
    k4a_playback_t handle_ = nullptr;
    k4a_transformation_t transformation_ = nullptr;
    MKVMetadata metadata_;

    std::string GetTagInMetadata(const std::string &tag_name);
};
}  // namespace io
}  // namespace open3d
