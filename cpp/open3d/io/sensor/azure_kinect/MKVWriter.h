// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/geometry/RGBDImage.h"
#include "open3d/io/sensor/azure_kinect/MKVMetadata.h"
#include "open3d/utility/IJsonConvertible.h"

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
