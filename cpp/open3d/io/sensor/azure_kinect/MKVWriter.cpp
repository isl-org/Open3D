// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/sensor/azure_kinect/MKVWriter.h"

#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>

#include "open3d/io/sensor/azure_kinect/K4aPlugin.h"

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
        utility::LogWarning("Unable to open file {}", filename);
        return false;
    }

    return true;
}

bool MKVWriter::IsOpened() { return handle_ != nullptr; }

bool MKVWriter::SetMetadata(const MKVMetadata &metadata) {
    metadata_ = metadata;

    if (K4A_RESULT_SUCCEEDED != k4a_plugin::k4a_record_write_header(handle_)) {
        utility::LogWarning("Unable to write header");
        return false;
    }
    return true;
}

void MKVWriter::Close() {
    if (K4A_RESULT_SUCCEEDED != k4a_plugin::k4a_record_flush(handle_)) {
        utility::LogWarning("Unable to flush before writing");
    }
    k4a_plugin::k4a_record_close(handle_);
}

bool MKVWriter::NextFrame(k4a_capture_t capture) {
    if (!IsOpened()) {
        utility::LogWarning("Null file handler. Please call Open().");
        return false;
    }

    if (K4A_RESULT_SUCCEEDED !=
        k4a_plugin::k4a_record_write_capture(handle_, capture)) {
        utility::LogWarning("Unable to write frame to mkv.");
        return false;
    }

    return true;
}
}  // namespace io
}  // namespace open3d
