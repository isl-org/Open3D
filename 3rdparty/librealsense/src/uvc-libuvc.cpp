// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#ifdef RS_USE_LIBUVC_BACKEND

//#define ENABLE_DEBUG_SPAM

#include "uvc.h"
#include "libuvc/libuvc.h"
#include "libuvc/libuvc_internal.h" // For LibUSB punchthrough
#include <thread>

namespace rsimpl
{
    namespace uvc
    {
        static void check(const char * call, uvc_error_t status)
        {
            if (status < 0) throw std::runtime_error(to_string() << call << "(...) returned " << uvc_strerror(status));
        }
        #define CALL_UVC(name, ...) check(#name, name(__VA_ARGS__))

        struct context
        {
            uvc_context_t * ctx;

            context() : ctx() { check("uvc_init", uvc_init(&ctx, nullptr)); }
            ~context() { if(ctx) uvc_exit(ctx); }
        };

        struct subdevice
        {
            uvc_device_handle_t * handle = nullptr;
            uvc_stream_ctrl_t ctrl;
            uint8_t unit;
            std::function<void(const void * frame)> callback;
        };

        struct device
        {
            const std::shared_ptr<context> parent;
            uvc_device_t * uvcdevice;
            int vid, pid;
            std::vector<subdevice> subdevices;
            std::vector<int> claimed_interfaces;

            device(std::shared_ptr<context> parent, uvc_device_t * uvcdevice) : parent(parent), uvcdevice(uvcdevice)
            {
                get_subdevice(0);
                
                uvc_device_descriptor_t * desc;
                CALL_UVC(uvc_get_device_descriptor, uvcdevice, &desc);
                vid = desc->idVendor;
                pid = desc->idProduct;
                uvc_free_device_descriptor(desc);
            }
            ~device()
            {
                for(auto interface_number : claimed_interfaces)
                {
                    int status = libusb_release_interface(get_subdevice(0).handle->usb_devh, interface_number);
                    if(status < 0) LOG_ERROR("libusb_release_interface(...) returned " << libusb_error_name(status));
                }

                for(auto & sub : subdevices) if(sub.handle) uvc_close(sub.handle);
                if(claimed_interfaces.size()) if(uvcdevice) uvc_unref_device(uvcdevice);
            }

            subdevice & get_subdevice(int subdevice_index)
            {
                if(subdevice_index >= subdevices.size()) subdevices.resize(subdevice_index+1);
                if(!subdevices[subdevice_index].handle) check("uvc_open2", uvc_open2(uvcdevice, &subdevices[subdevice_index].handle, subdevice_index));
                return subdevices[subdevice_index];
            }
        };

        ////////////
        // device //
        ////////////

        int get_vendor_id(const device & device) { return device.vid; }
        int get_product_id(const device & device) { return device.pid; }

        void get_control(const device & dev, const extension_unit & xu, uint8_t ctrl, void * data, int len)
        {
            int status = uvc_get_ctrl(const_cast<device &>(dev).get_subdevice(xu.subdevice).handle, xu.unit, ctrl, data, len, UVC_GET_CUR);
            if(status < 0) throw std::runtime_error(to_string() << "uvc_get_ctrl(...) returned " << libusb_error_name(status));
        }

        void set_control(device & device, const extension_unit & xu, uint8_t ctrl, void * data, int len)
        {
            int status = uvc_set_ctrl(device.get_subdevice(xu.subdevice).handle, xu.unit, ctrl, data, len);
            if(status < 0) throw std::runtime_error(to_string() << "uvc_set_ctrl(...) returned " << libusb_error_name(status));
        }

        void claim_interface(device & device, const guid & interface_guid, int interface_number)
        {
            int status = libusb_claim_interface(device.get_subdevice(0).handle->usb_devh, interface_number);
            if(status < 0) throw std::runtime_error(to_string() << "libusb_claim_interface(...) returned " << libusb_error_name(status));
            device.claimed_interfaces.push_back(interface_number);
        }

        void bulk_transfer(device & device, unsigned char endpoint, void * data, int length, int *actual_length, unsigned int timeout)
        {
            int status = libusb_bulk_transfer(device.get_subdevice(0).handle->usb_devh, endpoint, (unsigned char *)data, length, actual_length, timeout);
            if(status < 0) throw std::runtime_error(to_string() << "libusb_bulk_transfer(...) returned " << libusb_error_name(status));
        }

        void set_subdevice_mode(device & device, int subdevice_index, int width, int height, uint32_t fourcc, int fps, std::function<void(const void * frame)> callback)
        {
            auto & sub = device.get_subdevice(subdevice_index);
            check("get_stream_ctrl_format_size", uvc_get_stream_ctrl_format_size(sub.handle, &sub.ctrl, reinterpret_cast<const big_endian<uint32_t> &>(fourcc), width, height, fps));
            sub.callback = callback;
        }

        void start_streaming(device & device, int num_transfer_bufs)
        {
            for(auto & sub : device.subdevices)
            {
                if(sub.callback)
                {
                    #if defined (ENABLE_DEBUG_SPAM)
                    uvc_print_stream_ctrl(&sub.ctrl, stdout);
                    #endif

                    check("uvc_start_streaming", uvc_start_streaming(sub.handle, &sub.ctrl, [](uvc_frame * frame, void * user)
                    {
                        reinterpret_cast<subdevice *>(user)->callback(frame->data);
                    }, &sub, 0, num_transfer_bufs));
                }
            }
        }

        void stop_streaming(device & device)
        {
            // Stop all streaming
            for(auto & sub : device.subdevices)
            {
                if(sub.handle) uvc_stop_streaming(sub.handle);
                sub.ctrl = {};
                sub.callback = {};
            }
        }

        template<class T> void set_pu(uvc_device_handle_t * devh, int subdevice, uint8_t unit, uint8_t control, int value)
        {
            const int REQ_TYPE_SET = 0x21;
            unsigned char buffer[4];
            if(sizeof(T)==1) buffer[0] = value;
            if(sizeof(T)==2) SHORT_TO_SW(value, buffer);
            if(sizeof(T)==4) INT_TO_DW(value, buffer);
            int status = libusb_control_transfer(devh->usb_devh, REQ_TYPE_SET, UVC_SET_CUR, control << 8, unit << 8 | (subdevice*2), buffer, sizeof(T), 0);
            if(status < 0) throw std::runtime_error(to_string() << "libusb_control_transfer(...) returned " << libusb_error_name(status));
            if(status != sizeof(T)) throw std::runtime_error("insufficient data written to usb");
        }

        template<class T> int get_pu(uvc_device_handle_t * devh, int subdevice, uint8_t unit, uint8_t control, int uvc_get_thing)
        {
            const int REQ_TYPE_GET = 0xa1;
            unsigned char buffer[4];
            int status = libusb_control_transfer(devh->usb_devh, REQ_TYPE_GET, uvc_get_thing, control << 8, unit << 8 | (subdevice*2), buffer, sizeof(T), 0);
            if(status < 0) throw std::runtime_error(to_string() << "libusb_control_transfer(...) returned " << libusb_error_name(status));
            if(status != sizeof(T)) throw std::runtime_error("insufficient data read from usb");
            if(sizeof(T)==1) return buffer[0];
            if(sizeof(T)==2) return SW_TO_SHORT(buffer);
            if(sizeof(T)==4) return DW_TO_INT(buffer);
        }
        
        template<class T> void get_pu_range(uvc_device_handle_t * devh, int subdevice, uint8_t unit, uint8_t control, int * min, int * max)
        {
            if(min) *min = get_pu<T>(devh, subdevice, unit, control, UVC_GET_MIN);
            if(max) *max = get_pu<T>(devh, subdevice, unit, control, UVC_GET_MAX);
        }

        void get_pu_control_range(const device & device, int subdevice, rs_option option, int * min, int * max)
        {
            auto handle = const_cast<uvc::device &>(device).get_subdevice(subdevice).handle;
            int ct_unit = 0, pu_unit = 0;
            for(auto ct = uvc_get_input_terminals(handle); ct; ct = ct->next) ct_unit = ct->bTerminalID; // todo - Check supported caps
            for(auto pu = uvc_get_processing_units(handle); pu; pu = pu->next) pu_unit = pu->bUnitID; // todo - Check supported caps
            
            switch(option)
            {
            case RS_OPTION_COLOR_BACKLIGHT_COMPENSATION: return get_pu_range<uint16_t>(handle, subdevice, pu_unit, UVC_PU_BACKLIGHT_COMPENSATION_CONTROL, min, max);
            case RS_OPTION_COLOR_BRIGHTNESS: return get_pu_range<int16_t>(handle, subdevice, pu_unit, UVC_PU_BRIGHTNESS_CONTROL, min, max);
            case RS_OPTION_COLOR_CONTRAST: return get_pu_range<uint16_t>(handle, subdevice, pu_unit, UVC_PU_CONTRAST_CONTROL, min, max);
            case RS_OPTION_COLOR_EXPOSURE: return get_pu_range<uint32_t>(handle, subdevice, ct_unit, UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL, min, max);
            case RS_OPTION_COLOR_GAIN: return get_pu_range<uint16_t>(handle, subdevice, pu_unit, UVC_PU_GAIN_CONTROL, min, max);
            case RS_OPTION_COLOR_GAMMA: return get_pu_range<uint16_t>(handle, subdevice, pu_unit, UVC_PU_GAMMA_CONTROL, min, max);
            case RS_OPTION_COLOR_HUE: if(min) *min = 0; if(max) *max = 0; return; //return get_pu_range<int16_t>(handle, subdevice, pu_unit, UVC_PU_HUE_CONTROL, min, max);
            case RS_OPTION_COLOR_SATURATION: return get_pu_range<uint16_t>(handle, subdevice, pu_unit, UVC_PU_SATURATION_CONTROL, min, max);
            case RS_OPTION_COLOR_SHARPNESS: return get_pu_range<uint16_t>(handle, subdevice, pu_unit, UVC_PU_SHARPNESS_CONTROL, min, max);
            case RS_OPTION_COLOR_WHITE_BALANCE: return get_pu_range<uint16_t>(handle, subdevice, pu_unit, UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL, min, max);
            case RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE: if(min) *min = 0; if(max) *max = 1; return; // The next 2 options do not support range operations
            case RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE: if(min) *min = 0; if(max) *max = 1; return;
            default: throw std::logic_error("invalid option");
            }
        }
        
        void set_pu_control(device & device, int subdevice, rs_option option, int value)
        {            
            auto handle = device.get_subdevice(subdevice).handle;
            int ct_unit = 0, pu_unit = 0;
            for(auto ct = uvc_get_input_terminals(handle); ct; ct = ct->next) ct_unit = ct->bTerminalID; // todo - Check supported caps
            for(auto pu = uvc_get_processing_units(handle); pu; pu = pu->next) pu_unit = pu->bUnitID; // todo - Check supported caps

            switch(option)
            {
            case RS_OPTION_COLOR_BACKLIGHT_COMPENSATION: return set_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_BACKLIGHT_COMPENSATION_CONTROL, value);
            case RS_OPTION_COLOR_BRIGHTNESS: return set_pu<int16_t>(handle, subdevice, pu_unit, UVC_PU_BRIGHTNESS_CONTROL, value);
            case RS_OPTION_COLOR_CONTRAST: return set_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_CONTRAST_CONTROL, value);
            case RS_OPTION_COLOR_EXPOSURE: return set_pu<uint32_t>(handle, subdevice, ct_unit, UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL, value);
            case RS_OPTION_COLOR_GAIN: return set_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_GAIN_CONTROL, value);
            case RS_OPTION_COLOR_GAMMA: return set_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_GAMMA_CONTROL, value);
            case RS_OPTION_COLOR_HUE: return; // set_pu<int16_t>(handle, subdevice, pu_unit, UVC_PU_HUE_CONTROL, value); // Causes LIBUSB_ERROR_PIPE, may be related to not being able to set UVC_PU_HUE_AUTO_CONTROL
            case RS_OPTION_COLOR_SATURATION: return set_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_SATURATION_CONTROL, value);
            case RS_OPTION_COLOR_SHARPNESS: return set_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_SHARPNESS_CONTROL, value);
            case RS_OPTION_COLOR_WHITE_BALANCE: return set_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL, value);
            case RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE: return set_pu<uint8_t>(handle, subdevice, ct_unit, UVC_CT_AE_MODE_CONTROL, value ? 2 : 1); // Modes - (1: manual) (2: auto) (4: shutter priority) (8: aperture priority)
            case RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE: return set_pu<uint8_t>(handle, subdevice, pu_unit, UVC_PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL, value);
            default: throw std::logic_error("invalid option");
            }
        }

        int get_pu_control(const device & device, int subdevice, rs_option option)
        {
            auto handle = const_cast<uvc::device &>(device).get_subdevice(subdevice).handle;
            int ct_unit = 0, pu_unit = 0;
            for(auto ct = uvc_get_input_terminals(handle); ct; ct = ct->next) ct_unit = ct->bTerminalID; // todo - Check supported caps
            for(auto pu = uvc_get_processing_units(handle); pu; pu = pu->next) pu_unit = pu->bUnitID; // todo - Check supported caps

            switch(option)
            {
            case RS_OPTION_COLOR_BACKLIGHT_COMPENSATION: return get_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_BACKLIGHT_COMPENSATION_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_BRIGHTNESS: return get_pu<int16_t>(handle, subdevice, pu_unit, UVC_PU_BRIGHTNESS_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_CONTRAST: return get_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_CONTRAST_CONTROL,UVC_GET_CUR);
            case RS_OPTION_COLOR_EXPOSURE: return get_pu<uint32_t>(handle, subdevice, ct_unit, UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_GAIN: return get_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_GAIN_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_GAMMA: return get_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_GAMMA_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_HUE: return 0; //get_pu<int16_t>(handle, subdevice, pu_unit, UVC_PU_HUE_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_SATURATION: return get_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_SATURATION_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_SHARPNESS: return get_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_SHARPNESS_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_WHITE_BALANCE: return get_pu<uint16_t>(handle, subdevice, pu_unit, UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL, UVC_GET_CUR);
            case RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE: return get_pu<uint8_t>(handle, subdevice, ct_unit, UVC_CT_AE_MODE_CONTROL, UVC_GET_CUR) > 1; // Modes - (1: manual) (2: auto) (4: shutter priority) (8: aperture priority)
            case RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE: return get_pu<uint8_t>(handle, subdevice, pu_unit, UVC_PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL, UVC_GET_CUR);
            default: throw std::logic_error("invalid option");
            }
        }

        /////////////
        // context //
        /////////////

        std::shared_ptr<context> create_context()
        {
            return std::make_shared<context>();
        }

        std::vector<std::shared_ptr<device>> query_devices(std::shared_ptr<context> context)
        {
            std::vector<std::shared_ptr<device>> devices;
            
            uvc_device_t ** list;
            CALL_UVC(uvc_get_device_list, context->ctx, &list);
            for(auto it = list; *it; ++it) devices.push_back(std::make_shared<device>(context, *it));
            uvc_free_device_list(list, 1);
            return devices;
        }
    }
}

#endif