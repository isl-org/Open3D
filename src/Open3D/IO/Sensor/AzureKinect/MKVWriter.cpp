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

#include "Open3D/IO/Sensor/AzureKinect/MKVWriter.h"

#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>

#include "Open3D/IO/Sensor/AzureKinect/K4aPlugin.h"

namespace open3d {
namespace io {

MKVWriter::MKVWriter() : handle_(nullptr) {}

bool MKVWriter::Open(const std::string &filename,
                     const _k4a_device_configuration_t &config,
                     k4a_device_t device) {
    if (IsOpened()) {
        Close();
    }

    if (K4A_RESULT_SUCCEEDED != k4a_plugin::k4a_record_create(filename.c_str(),
                                                              device, config,
                                                              &handle_)) {
        utility::LogError("Unable to open file {}\n", filename);
        return false;
    }

    return true;
}

bool MKVWriter::IsOpened() { return handle_ != nullptr; }

bool MKVWriter::SetMetadata(const MKVMetadata &metadata) {
    metadata_ = metadata;

    if (K4A_RESULT_SUCCEEDED != k4a_plugin::k4a_record_write_header(handle_)) {
        utility::LogError("Unable to write header\n");
        return false;
    }
    return true;
}

void MKVWriter::Close() {
    if (K4A_RESULT_SUCCEEDED != k4a_plugin::k4a_record_flush(handle_)) {
        utility::LogError("Unable to flush before writing\n");
    }
    k4a_plugin::k4a_record_close(handle_);
}

bool MKVWriter::NextFrame(k4a_capture_t capture) {
    if (!IsOpened()) {
        utility::LogError("Null file handler. Please call Open().\n");
        return false;
    }

    if (K4A_RESULT_SUCCEEDED !=
        k4a_plugin::k4a_record_write_capture(handle_, capture)) {
        utility::LogError("Unable to write frame to mkv.\n");
        return false;
    }

    return true;
}
}  // namespace io
}  // namespace open3d
