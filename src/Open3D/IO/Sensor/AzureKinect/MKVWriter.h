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

#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/IO/Sensor/AzureKinect/MKVMetadata.h"
#include "Open3D/Utility/IJsonConvertible.h"

struct _k4a_device_configuration_t;  // Alias of k4a_device_configuration_t
struct _k4a_device_t;                // typedef _k4a_device_t* k4a_device_t;
struct _k4a_capture_t;               // typedef _k4a_capture_t* k4a_capture_t;
struct _k4a_record_t;                // typedef _k4a_record_t* k4a_record_t;

namespace open3d {
namespace io {

class MKVWriter {
public:
    MKVWriter();
    virtual ~MKVWriter() {}

    bool IsOpened();

    /* We assume device is already set properly according to config */
    bool Open(const std::string &filename,
              const _k4a_device_configuration_t &config,
              _k4a_device_t *device);
    void Close();

    bool SetMetadata(const MKVMetadata &metadata);
    bool NextFrame(_k4a_capture_t *);

private:
    _k4a_record_t *handle_;
    MKVMetadata metadata_;
};
}  // namespace io
}  // namespace open3d
