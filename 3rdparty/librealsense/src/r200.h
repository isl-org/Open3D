// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once
#ifndef LIBREALSENSE_R200_H
#define LIBREALSENSE_R200_H

#include "device.h"

namespace rsimpl
{
    class r200_camera final : public rs_device
    {
        bool is_disparity_mode_enabled() const;
        void on_update_depth_units(uint32_t units);
        void on_update_disparity_multiplier(double multiplier);
        uint32_t get_lr_framerate() const;
    public:
        r200_camera(std::shared_ptr<uvc::device> device, const static_device_info & info);
        ~r200_camera();

        bool supports_option(rs_option option) const override;
        void get_option_range(rs_option option, double & min, double & max, double & step) override;
        void set_options(const rs_option options[], int count, const double values[]) override;
        void get_options(const rs_option options[], int count, double values[]) override;

        void on_before_start(const std::vector<subdevice_mode_selection> & selected_modes) override;
        rs_stream select_key_stream(const std::vector<rsimpl::subdevice_mode_selection> & selected_modes) override;
        std::shared_ptr<frame_timestamp_reader> create_frame_timestamp_reader() const override;
    };

    std::shared_ptr<rs_device> make_r200_device(std::shared_ptr<uvc::device> device);
}

#endif
