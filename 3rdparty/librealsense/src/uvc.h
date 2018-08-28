// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once
#ifndef LIBREALSENSE_UVC_H
#define LIBREALSENSE_UVC_H

#include "types.h"

#include <memory>       // For shared_ptr
#include <functional>   // For function
#include <thread>       // For this_thread::sleep_for

namespace rsimpl
{
    namespace uvc
    {
        struct guid { uint32_t data1; uint16_t data2, data3; uint8_t data4[8]; };
        struct extension_unit { int subdevice, unit, node; guid id; };

        struct context; // Opaque type representing access to the underlying UVC implementation
        struct device; // Opaque type representing access to a specific UVC device

        // Enumerate devices
        std::shared_ptr<context> create_context();
        std::vector<std::shared_ptr<device>> query_devices(std::shared_ptr<context> context);

        // Static device properties
        int get_vendor_id(const device & device);
        int get_product_id(const device & device);

        // Direct USB controls
        void claim_interface(device & device, const guid & interface_guid, int interface_number);
        void bulk_transfer(device & device, unsigned char endpoint, void * data, int length, int *actual_length, unsigned int timeout);

        // Access CT and PU controls
        inline bool is_pu_control(rs_option option) { return option >= RS_OPTION_COLOR_BACKLIGHT_COMPENSATION && option <= RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE; }
        void get_pu_control_range(const device & device, int subdevice, rs_option option, int * min, int * max);
        void set_pu_control(device & device, int subdevice, rs_option option, int value);
        int get_pu_control(const device & device, int subdevice, rs_option option);

        // Access XU controls
        void set_control(device & device, const extension_unit & xu, uint8_t ctrl, void * data, int len);
        void get_control(const device & device, const extension_unit & xu, uint8_t ctrl, void * data, int len);

        // Control streaming
        void set_subdevice_mode(device & device, int subdevice_index, int width, int height, uint32_t fourcc, int fps, std::function<void(const void * frame)> callback);
        void start_streaming(device & device, int num_transfer_bufs);
        void stop_streaming(device & device);
        
        // Access CT, PU, and XU controls, and retry if failure occurs
        inline void set_pu_control_with_retry(device & device, int subdevice, rs_option option, int value)
        {
            // Try writing a control, if it fails, retry several times
            // TODO: We may wish to tune the retry counts and sleep times based on camera, platform, firmware, etc.
            for(int i=0; i<20; ++i)
            {
                try { set_pu_control(device, subdevice, option, value); return; }
                catch(...) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
            }
            set_pu_control(device, subdevice, option, value);
        }
        
        inline int get_pu_control_with_retry(const device & device, int subdevice, rs_option option)
        {
            // Try reading a control, if it fails, retry several times
            for(int i=0; i<20; ++i)
            {
                try { return get_pu_control(device, subdevice, option); }
                catch(...) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
            }
            return get_pu_control(device, subdevice, option);
        }
        
        inline void set_control_with_retry(device & device, const extension_unit & xu, uint8_t ctrl, void * data, int len)
        {
            // Try writing a control, if it fails, retry several times
            for(int i=0; i<20; ++i)
            {
                try { set_control(device, xu, ctrl, data, len); return; }
                catch(...) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
            }
            set_control(device, xu, ctrl, data, len);
        }
        
        inline void get_control_with_retry(const device & device, const extension_unit & xu, uint8_t ctrl, void * data, int len)
        {
            // Try reading a control, if it fails, retry several times
            for(int i=0; i<20; ++i)
            {
                try { get_control(device, xu, ctrl, data, len); return; }
                catch(...) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
            }
            get_control(device, xu, ctrl, data, len);
        }
    }
}

#endif
