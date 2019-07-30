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

#include <Open3D/Geometry/RGBDImage.h>
#include <Open3D/Utility/IJsonConvertible.h>

#include <k4a/k4a.h>
#include <k4arecord/record.h>

#include <json/json.h>
#include "MKVMetadata.h"

namespace open3d {

class MKVWriter {
public:
    bool IsOpened() { return handle_ != nullptr; }

    /* We assume device is already set properly according to config */
    int Open(const std::string &filename,
             const k4a_device_configuration_t &config,
             k4a_device_t device);
    void Close();

    int SetMetadata(const MKVMetadata& metadata);
    int NextFrame(k4a_capture_t);

private:
    k4a_record_t handle_ = nullptr;
    MKVMetadata metadata_;
};
}  // namespace open3d
