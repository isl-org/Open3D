// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "types.h"
#include "image.h"
#include "device.h"

#include <cstring>
#include <algorithm>
#include <array>

namespace rsimpl
{
    const char * get_string(rs_stream value)
    {
        #define CASE(X) case RS_STREAM_##X: return #X;
        switch(value)
        {
        CASE(DEPTH)
        CASE(COLOR)
        CASE(INFRARED)
        CASE(INFRARED2)
        CASE(POINTS)
        CASE(RECTIFIED_COLOR)
        CASE(COLOR_ALIGNED_TO_DEPTH)
        CASE(DEPTH_ALIGNED_TO_COLOR)
        CASE(DEPTH_ALIGNED_TO_RECTIFIED_COLOR)
        CASE(INFRARED2_ALIGNED_TO_DEPTH)
        CASE(DEPTH_ALIGNED_TO_INFRARED2)
        default: assert(!is_valid(value)); return nullptr;
        }
        #undef CASE
    }

    const char * get_string(rs_format value)
    {
        #define CASE(X) case RS_FORMAT_##X: return #X;
        switch(value)
        {
        CASE(ANY)
        CASE(Z16)
        CASE(DISPARITY16)
        CASE(XYZ32F)
        CASE(YUYV)
        CASE(RGB8)
        CASE(BGR8)
        CASE(RGBA8)
        CASE(BGRA8)
        CASE(Y8)
        CASE(Y16)
        CASE(RAW10)
        default: assert(!is_valid(value)); return nullptr;
        }
        #undef CASE
    }

    const char * get_string(rs_preset value)
    {
        #define CASE(X) case RS_PRESET_##X: return #X;
        switch(value)
        {
        CASE(BEST_QUALITY)
        CASE(LARGEST_IMAGE)
        CASE(HIGHEST_FRAMERATE)
        default: assert(!is_valid(value)); return nullptr;
        }
        #undef CASE
    }

    const char * get_string(rs_distortion value)
    {
        #define CASE(X) case RS_DISTORTION_##X: return #X;
        switch(value)
        {
        CASE(NONE)
        CASE(MODIFIED_BROWN_CONRADY)
        CASE(INVERSE_BROWN_CONRADY)
        default: assert(!is_valid(value)); return nullptr;
        }
        #undef CASE
    }

    const char * get_string(rs_option value)
    {
        #define CASE(X) case RS_OPTION_##X: return #X;
        switch(value)
        {
        CASE(COLOR_BACKLIGHT_COMPENSATION)
        CASE(COLOR_BRIGHTNESS)
        CASE(COLOR_CONTRAST)
        CASE(COLOR_EXPOSURE)
        CASE(COLOR_GAIN)
        CASE(COLOR_GAMMA)
        CASE(COLOR_HUE)
        CASE(COLOR_SATURATION)
        CASE(COLOR_SHARPNESS)
        CASE(COLOR_WHITE_BALANCE)
        CASE(COLOR_ENABLE_AUTO_EXPOSURE)
        CASE(COLOR_ENABLE_AUTO_WHITE_BALANCE)
        CASE(F200_LASER_POWER)
        CASE(F200_ACCURACY)
        CASE(F200_MOTION_RANGE)
        CASE(F200_FILTER_OPTION)
        CASE(F200_CONFIDENCE_THRESHOLD)
        CASE(SR300_DYNAMIC_FPS)
        CASE(SR300_AUTO_RANGE_ENABLE_MOTION_VERSUS_RANGE) 
        CASE(SR300_AUTO_RANGE_ENABLE_LASER)               
        CASE(SR300_AUTO_RANGE_MIN_MOTION_VERSUS_RANGE)    
        CASE(SR300_AUTO_RANGE_MAX_MOTION_VERSUS_RANGE)    
        CASE(SR300_AUTO_RANGE_START_MOTION_VERSUS_RANGE)  
        CASE(SR300_AUTO_RANGE_MIN_LASER)                  
        CASE(SR300_AUTO_RANGE_MAX_LASER)                  
        CASE(SR300_AUTO_RANGE_START_LASER)                
        CASE(SR300_AUTO_RANGE_UPPER_THRESHOLD) 
        CASE(SR300_AUTO_RANGE_LOWER_THRESHOLD) 
        CASE(R200_LR_AUTO_EXPOSURE_ENABLED)
        CASE(R200_LR_GAIN)
        CASE(R200_LR_EXPOSURE)
        CASE(R200_EMITTER_ENABLED)
        CASE(R200_DEPTH_UNITS)
        CASE(R200_DEPTH_CLAMP_MIN)
        CASE(R200_DEPTH_CLAMP_MAX)
        CASE(R200_DISPARITY_MULTIPLIER)
        CASE(R200_DISPARITY_SHIFT)
        CASE(R200_AUTO_EXPOSURE_MEAN_INTENSITY_SET_POINT)
        CASE(R200_AUTO_EXPOSURE_BRIGHT_RATIO_SET_POINT)  
        CASE(R200_AUTO_EXPOSURE_KP_GAIN)                 
        CASE(R200_AUTO_EXPOSURE_KP_EXPOSURE)             
        CASE(R200_AUTO_EXPOSURE_KP_DARK_THRESHOLD)       
        CASE(R200_AUTO_EXPOSURE_TOP_EDGE)       
        CASE(R200_AUTO_EXPOSURE_BOTTOM_EDGE)    
        CASE(R200_AUTO_EXPOSURE_LEFT_EDGE)      
        CASE(R200_AUTO_EXPOSURE_RIGHT_EDGE)     
        CASE(R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_DECREMENT)   
        CASE(R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_INCREMENT)   
        CASE(R200_DEPTH_CONTROL_MEDIAN_THRESHOLD)            
        CASE(R200_DEPTH_CONTROL_SCORE_MINIMUM_THRESHOLD)     
        CASE(R200_DEPTH_CONTROL_SCORE_MAXIMUM_THRESHOLD)     
        CASE(R200_DEPTH_CONTROL_TEXTURE_COUNT_THRESHOLD)     
        CASE(R200_DEPTH_CONTROL_TEXTURE_DIFFERENCE_THRESHOLD)
        CASE(R200_DEPTH_CONTROL_SECOND_PEAK_THRESHOLD)       
        CASE(R200_DEPTH_CONTROL_NEIGHBOR_THRESHOLD)          
        CASE(R200_DEPTH_CONTROL_LR_THRESHOLD)
        CASE(SR300_WAKEUP_DEV_PHASE1_PERIOD)
        CASE(SR300_WAKEUP_DEV_PHASE1_FPS)
        CASE(SR300_WAKEUP_DEV_PHASE2_PERIOD)
        CASE(SR300_WAKEUP_DEV_PHASE2_FPS)
        CASE(SR300_WAKEUP_DEV_RESET)
        CASE(SR300_WAKE_ON_USB_REASON)
        CASE(SR300_WAKE_ON_USB_CONFIDENCE)

        default: assert(!is_valid(value)); return nullptr;
        }
        #undef CASE
    }

    size_t subdevice_mode_selection::get_image_size(rs_stream stream) const
    {
        return rsimpl::get_image_size(get_width(), get_height(), get_format(stream));
    }

    void subdevice_mode_selection::unpack(byte * const dest[], const byte * source) const
    {
        const int MAX_OUTPUTS = 2;
        const auto & outputs = get_outputs();        
        assert(outputs.size() <= MAX_OUTPUTS);

        // Determine input stride (and apply cropping)
        const byte * in = source;
        size_t in_stride = mode.pf.get_image_size(mode.native_dims.x, 1);
        if(pad_crop < 0) in += in_stride * -pad_crop + mode.pf.get_image_size(-pad_crop, 1);

        // Determine output stride (and apply padding)
        byte * out[MAX_OUTPUTS];
        size_t out_stride[MAX_OUTPUTS];
        for(size_t i=0; i<outputs.size(); ++i)
        {
            out[i] = dest[i];
            out_stride[i] = rsimpl::get_image_size(get_width(), 1, outputs[i].second);
            if(pad_crop > 0) out[i] += out_stride[i] * pad_crop + rsimpl::get_image_size(pad_crop, 1, outputs[i].second);
        }

        // Unpack (potentially a subrect of) the source image into (potentially a subrect of) the destination buffers
        const int unpack_width = std::min(mode.native_intrinsics.width, get_width()), unpack_height = std::min(mode.native_intrinsics.height, get_height());
        if(mode.native_dims.x == get_width())
        {
            // If not strided, unpack as though it were a single long row
            mode.pf.unpackers[unpacker_index].unpack(out, in, unpack_width * unpack_height);
        }
        else
        {
            // Otherwise unpack one row at a time
            assert(mode.pf.plane_count == 1); // Can't unpack planar formats row-by-row (at least not with the current architecture, would need to pass multiple source ptrs to unpack)
            for(int i=0; i<unpack_height; ++i)
            {
                mode.pf.unpackers[unpacker_index].unpack(out, in, unpack_width);
                for(size_t i=0; i<outputs.size(); ++i) out[i] += out_stride[i];
                in += in_stride;
            }
        }
    }

    ////////////////////////
    // static_device_info //
    ////////////////////////

    static_device_info::static_device_info()
    {
        for(auto & s : stream_subdevices) s = -1;
        for(auto & s : presets) for(auto & p : s) p = stream_request();
        for(auto & p : stream_poses)
        {
            p = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}};
        }
    }

    subdevice_mode_selection device_config::select_mode(const stream_request (& requests)[RS_STREAM_NATIVE_COUNT], int subdevice_index) const
    {
        // Determine if the user has requested any streams which are supplied by this subdevice
        bool any_stream_requested = false;
        std::array<bool, RS_STREAM_NATIVE_COUNT> stream_requested = {};
        for(int j = 0; j < RS_STREAM_NATIVE_COUNT; ++j)
        {
            if(requests[j].enabled && info.stream_subdevices[j] == subdevice_index)
            {
                stream_requested[j] = true;
                any_stream_requested = true;
            }
        }

        // If no streams were requested, skip to the next subdevice
        if(!any_stream_requested) return subdevice_mode_selection();

        // Look for an appropriate mode
        for(auto & subdevice_mode : info.subdevice_modes)
        {
            // Skip modes that apply to other subdevices
            if(subdevice_mode.subdevice != subdevice_index) continue;

            for(auto pad_crop : subdevice_mode.pad_crop_options)
            {
                for(auto & unpacker : subdevice_mode.pf.unpackers)
                {
                    auto selection = subdevice_mode_selection(subdevice_mode, pad_crop, &unpacker - subdevice_mode.pf.unpackers.data());

                    // Determine if this mode satisfies the requirements on our requested streams
                    auto stream_unsatisfied = stream_requested;
                    for(auto & output : unpacker.outputs)
                    {
                        const auto & req = requests[output.first];
                        if(req.enabled && (req.width == 0 || req.width == selection.get_width())
                                       && (req.height == 0 || req.height == selection.get_height())
                                       && (req.format == RS_FORMAT_ANY || req.format == selection.get_format(output.first))
                                       && (req.fps == 0 || req.fps == subdevice_mode.fps))
                        {
                            stream_unsatisfied[output.first] = false;
                        }
                    }

                    // If any requested streams are still unsatisfied, skip to the next mode
                    if(std::any_of(begin(stream_unsatisfied), end(stream_unsatisfied), [](bool b) { return b; })) continue;
                    return selection;
                }
            }
        }

        // If we did not find an appropriate mode, report an error
        std::ostringstream ss;
        ss << "uvc subdevice " << subdevice_index << " cannot provide";
        bool first = true;
        for(int j = 0; j < RS_STREAM_NATIVE_COUNT; ++j)
        {
            if(!stream_requested[j]) continue;
            ss << (first ? " " : " and ");
            ss << requests[j].width << 'x' << requests[j].height << ':' << get_string(requests[j].format);
            ss << '@' << requests[j].fps << "Hz " << get_string((rs_stream)j);
            first = false;
        }
        throw std::runtime_error(ss.str());
    }

    std::vector<subdevice_mode_selection> device_config::select_modes(const stream_request (&reqs)[RS_STREAM_NATIVE_COUNT]) const
    {
        // Make a mutable copy of our array
        stream_request requests[RS_STREAM_NATIVE_COUNT];
        for(int i=0; i<RS_STREAM_NATIVE_COUNT; ++i) requests[i] = reqs[i];

        // Check and modify requests to enforce all interstream constraints
        for(auto & rule : info.interstream_rules)
        {
            auto & a = requests[rule.a], & b = requests[rule.b]; auto f = rule.field;
            if(a.enabled && b.enabled)
            {
                // Check for incompatibility if both values specified
                if(a.*f != 0 && b.*f != 0 && a.*f + rule.delta != b.*f && a.*f + rule.delta2 != b.*f)
                {
                    throw std::runtime_error(to_string() << "requested " << rule.a << " and " << rule.b << " settings are incompatible");
                }

                // If only one value is specified, modify the other request to match
                if(a.*f != 0 && b.*f == 0) b.*f = a.*f + rule.delta;
                if(a.*f == 0 && b.*f != 0) a.*f = b.*f - rule.delta;
            }
        }

        // Select subdevice modes needed to satisfy our requests
        int num_subdevices = 0;
        for(auto & mode : info.subdevice_modes) num_subdevices = std::max(num_subdevices, mode.subdevice+1);
        std::vector<subdevice_mode_selection> selected_modes;
        for(int i = 0; i < num_subdevices; ++i)
        {
            auto selection = select_mode(requests, i);
            if(selection.mode.pf.fourcc) selected_modes.push_back(selection);
        }
        return selected_modes;
    }
}
