// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "r200.h"
#include "r200-private.h"
#include "image.h"

#include <cstring>
#include <climits>
#include <algorithm>

namespace rsimpl
{
    r200_camera::r200_camera(std::shared_ptr<uvc::device> device, const static_device_info & info) : rs_device(device, info)
    {
        rs_option opt[] = {RS_OPTION_R200_DEPTH_UNITS}; double units;
        get_options(opt, 1, &units);
        on_update_depth_units(static_cast<int>(units));
    }
    
    r200_camera::~r200_camera()
    {

    }

    std::shared_ptr<rs_device> make_r200_device(std::shared_ptr<uvc::device> device)
    {
        LOG_INFO("Connecting to Intel RealSense R200");
        
        static_device_info info;
        info.name = {"Intel RealSense R200"};
        info.stream_subdevices[RS_STREAM_DEPTH] = 1;
        info.stream_subdevices[RS_STREAM_COLOR] = 2;
        info.stream_subdevices[RS_STREAM_INFRARED ] = 0;
        info.stream_subdevices[RS_STREAM_INFRARED2] = 0;

        auto c = r200::read_camera_info(*device);
        
        // Set up modes for left/right/z images
        for(auto fps : {30, 60, 90})
        {
            // Subdevice 0 can provide left/right infrared via four pixel formats, in three resolutions, which can either be uncropped or cropped to match Z
            for(auto pf : {pf_y8, pf_y8i, pf_y16, pf_y12i})
            {
                info.subdevice_modes.push_back({0, {640, 481}, pf, fps, c.modesLR[0], {}, {0, -6}});  
                info.subdevice_modes.push_back({0, {640, 373}, pf, fps, c.modesLR[1], {}, {0, -6}});  
                info.subdevice_modes.push_back({0, {640, 254}, pf, fps, c.modesLR[2], {}, {0, -6}});  
            }

            // Subdevice 1 can provide depth, in three resolutions, which can either be unpadded or padded to match left/right
            info.subdevice_modes.push_back({1, {628, 469}, pf_z16,  fps, pad_crop_intrinsics(c.modesLR[0], -6), {}, {0, +6}});
            info.subdevice_modes.push_back({1, {628, 361}, pf_z16,  fps, pad_crop_intrinsics(c.modesLR[1], -6), {}, {0, +6}});
            info.subdevice_modes.push_back({1, {628, 242}, pf_z16,  fps, pad_crop_intrinsics(c.modesLR[2], -6), {}, {0, +6}});
        }

        // Subdevice 2 can provide color, in several formats and framerates
        info.subdevice_modes.push_back({2, { 320,  240}, pf_yuy2, 60, scale_intrinsics(c.intrinsicsThird[1], 320, 240), {scale_intrinsics(c.modesThird[1][0], 320, 240)}, {0}});
        info.subdevice_modes.push_back({2, { 320,  240}, pf_yuy2, 30, scale_intrinsics(c.intrinsicsThird[1], 320, 240), {scale_intrinsics(c.modesThird[1][0], 320, 240)}, {0}});
        info.subdevice_modes.push_back({2, { 640,  480}, pf_yuy2, 60, c.intrinsicsThird[1], {c.modesThird[1][0]}, {0}});
        info.subdevice_modes.push_back({2, { 640,  480}, pf_yuy2, 30, c.intrinsicsThird[1], {c.modesThird[1][0]}, {0}});
		info.subdevice_modes.push_back({2, {1920, 1080}, pf_yuy2, 15, c.intrinsicsThird[0], {c.modesThird[0][0]}, {0}});
        info.subdevice_modes.push_back({2, {1920, 1080}, pf_yuy2, 30, c.intrinsicsThird[0], {c.modesThird[0][0]}, {0}});
        info.subdevice_modes.push_back({2, {2400, 1081}, pf_rw10, 30, c.intrinsicsThird[0], {c.modesThird[0][0]}, {0}});

        // Set up interstream rules for left/right/z images
        for(auto ir : {RS_STREAM_INFRARED, RS_STREAM_INFRARED2})
        {
            info.interstream_rules.push_back({RS_STREAM_DEPTH, ir, &stream_request::width, 0, 12});
            info.interstream_rules.push_back({RS_STREAM_DEPTH, ir, &stream_request::height, 0, 12});
            info.interstream_rules.push_back({RS_STREAM_DEPTH, ir, &stream_request::fps, 0, 0});
        }

        info.presets[RS_STREAM_INFRARED][RS_PRESET_BEST_QUALITY] = {true, 480, 360, RS_FORMAT_Y8,   60};
        info.presets[RS_STREAM_DEPTH   ][RS_PRESET_BEST_QUALITY] = {true, 480, 360, RS_FORMAT_Z16,  60};
        info.presets[RS_STREAM_COLOR   ][RS_PRESET_BEST_QUALITY] = {true, 640, 480, RS_FORMAT_RGB8, 60};

        info.presets[RS_STREAM_INFRARED][RS_PRESET_LARGEST_IMAGE] = {true,  640,  480, RS_FORMAT_Y8,   60};
        info.presets[RS_STREAM_DEPTH   ][RS_PRESET_LARGEST_IMAGE] = {true,  640,  480, RS_FORMAT_Z16,  60};
        info.presets[RS_STREAM_COLOR   ][RS_PRESET_LARGEST_IMAGE] = {true, 1920, 1080, RS_FORMAT_RGB8, 30};

        info.presets[RS_STREAM_INFRARED][RS_PRESET_HIGHEST_FRAMERATE] = {true, 320, 240, RS_FORMAT_Y8,   90};
        info.presets[RS_STREAM_DEPTH   ][RS_PRESET_HIGHEST_FRAMERATE] = {true, 320, 240, RS_FORMAT_Z16,  90};
        info.presets[RS_STREAM_COLOR   ][RS_PRESET_HIGHEST_FRAMERATE] = {true, 640, 480, RS_FORMAT_RGB8, 60};

        for(int i=0; i<RS_PRESET_COUNT; ++i) 
			info.presets[RS_STREAM_INFRARED2][i] = info.presets[RS_STREAM_INFRARED][i];

        info.options = {
            {RS_OPTION_R200_LR_AUTO_EXPOSURE_ENABLED,                   0, 1,           1},
            {RS_OPTION_R200_EMITTER_ENABLED,                            0, 1,           1},
            {RS_OPTION_R200_DEPTH_UNITS,                                1, INT_MAX,     1}, // What is the real range?
            {RS_OPTION_R200_DEPTH_CLAMP_MIN,                            0, USHRT_MAX,   1},
            {RS_OPTION_R200_DEPTH_CLAMP_MAX,                            0, USHRT_MAX,   1},
            {RS_OPTION_R200_DISPARITY_MULTIPLIER,                       1, 1000,        1},
            {RS_OPTION_R200_DISPARITY_SHIFT,                            0, 0,           1},

            {RS_OPTION_R200_AUTO_EXPOSURE_MEAN_INTENSITY_SET_POINT,     0, 4095,        0},
            {RS_OPTION_R200_AUTO_EXPOSURE_BRIGHT_RATIO_SET_POINT,       0, 1,           0},
            {RS_OPTION_R200_AUTO_EXPOSURE_KP_GAIN,                      0, 1000,        0},
            {RS_OPTION_R200_AUTO_EXPOSURE_KP_EXPOSURE,                  0, 1000,        0},
            {RS_OPTION_R200_AUTO_EXPOSURE_KP_DARK_THRESHOLD,            0, 1000,        0},
            {RS_OPTION_R200_AUTO_EXPOSURE_TOP_EDGE,                     0, USHRT_MAX,   1},
            {RS_OPTION_R200_AUTO_EXPOSURE_BOTTOM_EDGE,                  0, USHRT_MAX,   1},
            {RS_OPTION_R200_AUTO_EXPOSURE_LEFT_EDGE,                    0, USHRT_MAX,   1},
            {RS_OPTION_R200_AUTO_EXPOSURE_RIGHT_EDGE,                   0, USHRT_MAX,   1},

            {RS_OPTION_R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_DECREMENT,    0, 0xFF,        1},
            {RS_OPTION_R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_INCREMENT,    0, 0xFF,        1},
            {RS_OPTION_R200_DEPTH_CONTROL_MEDIAN_THRESHOLD,             0, 0x3FF,       1},
            {RS_OPTION_R200_DEPTH_CONTROL_SCORE_MINIMUM_THRESHOLD,      0, 0x3FF,       1},
            {RS_OPTION_R200_DEPTH_CONTROL_SCORE_MAXIMUM_THRESHOLD,      0, 0x3FF,       1},
            {RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_COUNT_THRESHOLD,      0, 0x1F,        1},
            {RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_DIFFERENCE_THRESHOLD, 0, 0x3FF,       1},
            {RS_OPTION_R200_DEPTH_CONTROL_SECOND_PEAK_THRESHOLD,        0, 0x3FF,       1},
            {RS_OPTION_R200_DEPTH_CONTROL_NEIGHBOR_THRESHOLD,           0, 0x3FF,       1},
            {RS_OPTION_R200_DEPTH_CONTROL_LR_THRESHOLD,                 0, 0x7FF,       1},
        };

        // We select the depth/left infrared camera's viewpoint to be the origin
        info.stream_poses[RS_STREAM_DEPTH] = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}};
        info.stream_poses[RS_STREAM_INFRARED] = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}};

        // The right infrared camera is offset along the +x axis by the baseline (B)
        info.stream_poses[RS_STREAM_INFRARED2] = {{{1,0,0},{0,1,0},{0,0,1}}, {c.B * 0.001f, 0, 0}}; // Sterling comment

		// The transformation between the depth camera and third camera is described by a translation vector (T), followed by rotation matrix (Rthird)
        for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) 
			info.stream_poses[RS_STREAM_COLOR].orientation(i,j) = c.Rthird[i*3+j];
        for(int i=0; i<3; ++i) 
			info.stream_poses[RS_STREAM_COLOR].position[i] = c.T[i] * 0.001f;

        // Our position is added AFTER orientation is applied, not before, so we must multiply Rthird * T to compute it
        info.stream_poses[RS_STREAM_COLOR].position = info.stream_poses[RS_STREAM_COLOR].orientation * info.stream_poses[RS_STREAM_COLOR].position;
        info.nominal_depth_scale = 0.001f;
        info.serial = std::to_string(c.serial_number);
        info.firmware_version = r200::read_firmware_version(*device);

		// On LibUVC backends, the R200 should use four transfer buffers
        info.num_libuvc_transfer_buffers = 4;
        return std::make_shared<r200_camera>(device, info);
    }

    bool r200_camera::is_disparity_mode_enabled() const
    {
        auto & depth = get_stream_interface(RS_STREAM_DEPTH);
        return depth.is_enabled() && depth.get_format() == RS_FORMAT_DISPARITY16;
    }

    void r200_camera::on_update_depth_units(uint32_t units)
    {
        if(is_disparity_mode_enabled()) return;
        config.depth_scale = (float)units / 1000000; // Convert from micrometers to meters
    }

    void r200_camera::on_update_disparity_multiplier(double multiplier)
    {
        if(!is_disparity_mode_enabled()) return;
        auto & depth = get_stream_interface(RS_STREAM_DEPTH);
        float baseline = get_stream_interface(RS_STREAM_INFRARED2).get_extrinsics_to(depth).translation[0];
        config.depth_scale = static_cast<float>(depth.get_intrinsics().fx * baseline * multiplier);
    }

    void r200_camera::set_options(const rs_option options[], int count, const double values[])
    {
        auto & dev = get_device();
        auto minmax_writer = make_struct_interface<r200::range    >([&dev]() { return r200::get_min_max_depth(dev);           }, [&dev](r200::range     v) { r200::set_min_max_depth(dev,v);           });
        auto disp_writer   = make_struct_interface<r200::disp_mode>([&dev]() { return r200::get_disparity_mode(dev);          }, [&dev](r200::disp_mode v) { r200::set_disparity_mode(dev,v);          });
        auto ae_writer     = make_struct_interface<r200::ae_params>([&dev]() { return r200::get_lr_auto_exposure_params(dev); }, [&dev](r200::ae_params v) { r200::set_lr_auto_exposure_params(dev,v); });
        auto dc_writer     = make_struct_interface<r200::dc_params>([&dev]() { return r200::get_depth_params(dev);            }, [&dev](r200::dc_params v) { r200::set_depth_params(dev,v);            });

        for(int i=0; i<count; ++i)
        {
            if(uvc::is_pu_control(options[i]))
            {
                uvc::set_pu_control_with_retry(get_device(), 2, options[i], static_cast<int>(values[i]));
                continue;
            }

            switch(options[i])
            {
            case RS_OPTION_R200_LR_AUTO_EXPOSURE_ENABLED:                   r200::set_lr_exposure_mode(get_device(), static_cast<uint8_t>(values[i])); break;
            case RS_OPTION_R200_LR_GAIN:                                    r200::set_lr_gain(get_device(), {get_lr_framerate(), static_cast<uint32_t>(values[i])}); break; // TODO: May need to set this on start if framerate changes
            case RS_OPTION_R200_LR_EXPOSURE:                                r200::set_lr_exposure(get_device(), {get_lr_framerate(), static_cast<uint32_t>(values[i])}); break; // TODO: May need to set this on start if framerate changes
            case RS_OPTION_R200_EMITTER_ENABLED:                            r200::set_emitter_state(get_device(), !!values[i]); break;
            case RS_OPTION_R200_DEPTH_UNITS:                                r200::set_depth_units(get_device(), static_cast<uint32_t>(values[i])); 
                                                                            on_update_depth_units(static_cast<uint32_t>(values[i])); break;

            case RS_OPTION_R200_DEPTH_CLAMP_MIN:                            minmax_writer.set(&r200::range::min, values[i]); break;
            case RS_OPTION_R200_DEPTH_CLAMP_MAX:                            minmax_writer.set(&r200::range::max, values[i]); break;

            case RS_OPTION_R200_DISPARITY_MULTIPLIER:                       disp_writer.set(&r200::disp_mode::disparity_multiplier, values[i]); break;
            case RS_OPTION_R200_DISPARITY_SHIFT:                            r200::set_disparity_shift(get_device(), static_cast<uint32_t>(values[i])); break;

            case RS_OPTION_R200_AUTO_EXPOSURE_MEAN_INTENSITY_SET_POINT:     ae_writer.set(&r200::ae_params::mean_intensity_set_point, values[i]); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_BRIGHT_RATIO_SET_POINT:       ae_writer.set(&r200::ae_params::bright_ratio_set_point,   values[i]); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_KP_GAIN:                      ae_writer.set(&r200::ae_params::kp_gain,                  values[i]); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_KP_EXPOSURE:                  ae_writer.set(&r200::ae_params::kp_exposure,              values[i]); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_KP_DARK_THRESHOLD:            ae_writer.set(&r200::ae_params::kp_dark_threshold,        values[i]); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_TOP_EDGE:                     ae_writer.set(&r200::ae_params::exposure_top_edge,        values[i]); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_BOTTOM_EDGE:                  ae_writer.set(&r200::ae_params::exposure_bottom_edge,     values[i]); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_LEFT_EDGE:                    ae_writer.set(&r200::ae_params::exposure_left_edge,       values[i]); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_RIGHT_EDGE:                   ae_writer.set(&r200::ae_params::exposure_right_edge,      values[i]); break;

            case RS_OPTION_R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_DECREMENT:    dc_writer.set(&r200::dc_params::robbins_munroe_minus_inc, values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_INCREMENT:    dc_writer.set(&r200::dc_params::robbins_munroe_plus_inc,  values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_MEDIAN_THRESHOLD:             dc_writer.set(&r200::dc_params::median_thresh,            values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_SCORE_MINIMUM_THRESHOLD:      dc_writer.set(&r200::dc_params::score_min_thresh,         values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_SCORE_MAXIMUM_THRESHOLD:      dc_writer.set(&r200::dc_params::score_max_thresh,         values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_COUNT_THRESHOLD:      dc_writer.set(&r200::dc_params::texture_count_thresh,     values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_DIFFERENCE_THRESHOLD: dc_writer.set(&r200::dc_params::texture_diff_thresh,      values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_SECOND_PEAK_THRESHOLD:        dc_writer.set(&r200::dc_params::second_peak_thresh,       values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_NEIGHBOR_THRESHOLD:           dc_writer.set(&r200::dc_params::neighbor_thresh,          values[i]); break;
            case RS_OPTION_R200_DEPTH_CONTROL_LR_THRESHOLD:                 dc_writer.set(&r200::dc_params::lr_thresh,                values[i]); break;

            default: LOG_WARNING("Cannot set " << options[i] << " to " << values[i] << " on " << get_name()); break;
            }
        }

        minmax_writer.commit();
        disp_writer.commit();
        if(disp_writer.active) on_update_disparity_multiplier(disp_writer.struct_.disparity_multiplier);
        ae_writer.commit();
        dc_writer.commit();
    }

    void r200_camera::get_options(const rs_option options[], int count, double values[])
    {
        auto & dev = get_device();
        auto minmax_reader = make_struct_interface<r200::range    >([&dev]() { return r200::get_min_max_depth(dev);           }, [&dev](r200::range     v) { r200::set_min_max_depth(dev,v);           });
        auto disp_reader   = make_struct_interface<r200::disp_mode>([&dev]() { return r200::get_disparity_mode(dev);          }, [&dev](r200::disp_mode v) { r200::set_disparity_mode(dev,v);          });
        auto ae_reader     = make_struct_interface<r200::ae_params>([&dev]() { return r200::get_lr_auto_exposure_params(dev); }, [&dev](r200::ae_params v) { r200::set_lr_auto_exposure_params(dev,v); });
        auto dc_reader     = make_struct_interface<r200::dc_params>([&dev]() { return r200::get_depth_params(dev);            }, [&dev](r200::dc_params v) { r200::set_depth_params(dev,v);            }); 

        for(int i=0; i<count; ++i)
        {
            if(uvc::is_pu_control(options[i]))
            {
                values[i] = uvc::get_pu_control(get_device(), 2, options[i]);
                continue;
            }

            switch(options[i])
            {
            case RS_OPTION_R200_LR_AUTO_EXPOSURE_ENABLED:                   values[i] = r200::get_lr_exposure_mode(get_device()); break;
            
            case RS_OPTION_R200_LR_GAIN: // Gain is framerate dependent
                r200::set_lr_gain_discovery(get_device(), {get_lr_framerate()});
                values[i] = r200::get_lr_gain(get_device()).value;
                break;
            case RS_OPTION_R200_LR_EXPOSURE: // Exposure is framerate dependent
                r200::set_lr_exposure_discovery(get_device(), {get_lr_framerate()});
                values[i] = r200::get_lr_exposure(get_device()).value;
                break;
            case RS_OPTION_R200_EMITTER_ENABLED:
                values[i] = r200::get_emitter_state(get_device(), is_capturing(), get_stream_interface(RS_STREAM_DEPTH).is_enabled());
                break;

            case RS_OPTION_R200_DEPTH_UNITS:                                values[i] = r200::get_depth_units(get_device());  break;

            case RS_OPTION_R200_DEPTH_CLAMP_MIN:                            values[i] = minmax_reader.get(&r200::range::min); break;
            case RS_OPTION_R200_DEPTH_CLAMP_MAX:                            values[i] = minmax_reader.get(&r200::range::max); break;

            case RS_OPTION_R200_DISPARITY_MULTIPLIER:                       values[i] = disp_reader.get(&r200::disp_mode::disparity_multiplier); break;
            case RS_OPTION_R200_DISPARITY_SHIFT:                            values[i] = r200::get_disparity_shift(get_device()); break;

            case RS_OPTION_R200_AUTO_EXPOSURE_MEAN_INTENSITY_SET_POINT:     values[i] = ae_reader.get(&r200::ae_params::mean_intensity_set_point); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_BRIGHT_RATIO_SET_POINT:       values[i] = ae_reader.get(&r200::ae_params::bright_ratio_set_point  ); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_KP_GAIN:                      values[i] = ae_reader.get(&r200::ae_params::kp_gain                 ); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_KP_EXPOSURE:                  values[i] = ae_reader.get(&r200::ae_params::kp_exposure             ); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_KP_DARK_THRESHOLD:            values[i] = ae_reader.get(&r200::ae_params::kp_dark_threshold       ); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_TOP_EDGE:                     values[i] = ae_reader.get(&r200::ae_params::exposure_top_edge       ); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_BOTTOM_EDGE:                  values[i] = ae_reader.get(&r200::ae_params::exposure_bottom_edge    ); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_LEFT_EDGE:                    values[i] = ae_reader.get(&r200::ae_params::exposure_left_edge      ); break;
            case RS_OPTION_R200_AUTO_EXPOSURE_RIGHT_EDGE:                   values[i] = ae_reader.get(&r200::ae_params::exposure_right_edge     ); break;

            case RS_OPTION_R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_DECREMENT:    values[i] = dc_reader.get(&r200::dc_params::robbins_munroe_minus_inc); break;
            case RS_OPTION_R200_DEPTH_CONTROL_ESTIMATE_MEDIAN_INCREMENT:    values[i] = dc_reader.get(&r200::dc_params::robbins_munroe_plus_inc ); break;
            case RS_OPTION_R200_DEPTH_CONTROL_MEDIAN_THRESHOLD:             values[i] = dc_reader.get(&r200::dc_params::median_thresh           ); break;
            case RS_OPTION_R200_DEPTH_CONTROL_SCORE_MINIMUM_THRESHOLD:      values[i] = dc_reader.get(&r200::dc_params::score_min_thresh        ); break;
            case RS_OPTION_R200_DEPTH_CONTROL_SCORE_MAXIMUM_THRESHOLD:      values[i] = dc_reader.get(&r200::dc_params::score_max_thresh        ); break;
            case RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_COUNT_THRESHOLD:      values[i] = dc_reader.get(&r200::dc_params::texture_count_thresh    ); break;
            case RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_DIFFERENCE_THRESHOLD: values[i] = dc_reader.get(&r200::dc_params::texture_diff_thresh     ); break;
            case RS_OPTION_R200_DEPTH_CONTROL_SECOND_PEAK_THRESHOLD:        values[i] = dc_reader.get(&r200::dc_params::second_peak_thresh      ); break;
            case RS_OPTION_R200_DEPTH_CONTROL_NEIGHBOR_THRESHOLD:           values[i] = dc_reader.get(&r200::dc_params::neighbor_thresh         ); break;
            case RS_OPTION_R200_DEPTH_CONTROL_LR_THRESHOLD:                 values[i] = dc_reader.get(&r200::dc_params::lr_thresh               ); break;

            default: LOG_WARNING("Cannot get " << options[i] << " on " << get_name()); break;
            }
        }
    }

    void r200_camera::on_before_start(const std::vector<subdevice_mode_selection> & selected_modes)
    {
        rs_option depth_units_option = RS_OPTION_R200_DEPTH_UNITS;
        double depth_units;

        uint8_t streamIntent = 0;
        for(const auto & m : selected_modes)
        {
            switch(m.mode.subdevice)
            {
            case 0: streamIntent |= r200::STATUS_BIT_LR_STREAMING; break;
            case 2: streamIntent |= r200::STATUS_BIT_WEB_STREAMING; break;
            case 1: 
                streamIntent |= r200::STATUS_BIT_Z_STREAMING; 
                auto dm = r200::get_disparity_mode(get_device());
                switch(m.get_format(RS_STREAM_DEPTH))
                {
                default: throw std::logic_error("unsupported R200 depth format");
                case RS_FORMAT_Z16: 
                    dm.is_disparity_enabled = 0;
                    get_options(&depth_units_option, 1, &depth_units);
                    on_update_depth_units(static_cast<int>(depth_units));
                    break;
                case RS_FORMAT_DISPARITY16: 
                    dm.is_disparity_enabled = 1;
                    on_update_disparity_multiplier(static_cast<float>(dm.disparity_multiplier));
                    break;
                }
                r200::set_disparity_mode(get_device(), dm);
                break;
            }
        }
        r200::set_stream_intent(get_device(), streamIntent);
    }

    rs_stream r200_camera::select_key_stream(const std::vector<rsimpl::subdevice_mode_selection> & selected_modes)
    {
        // When all streams are enabled at an identical framerate, R200 images are delivered in the order: Z -> Third -> L/R
        // To maximize the chance of being able to deliver coherent framesets, we want to wait on the latest image coming from
        // a stream running at the fastest framerate.
        int fps[RS_STREAM_NATIVE_COUNT] = {}, max_fps = 0;
        for(const auto & m : selected_modes)
        {
            for(const auto & output : m.get_outputs())
            {
                fps[output.first] = m.mode.fps;
                max_fps = std::max(max_fps, m.mode.fps);
            }
        }

        // Select the "latest arriving" stream which is running at the fastest framerate
        for(auto s : {RS_STREAM_COLOR, RS_STREAM_INFRARED2, RS_STREAM_INFRARED})
        {
            if(fps[s] == max_fps) return s;
        }
        return RS_STREAM_DEPTH;
    }

    uint32_t r200_camera::get_lr_framerate() const
    {
        for(auto s : {RS_STREAM_DEPTH, RS_STREAM_INFRARED, RS_STREAM_INFRARED2})
        {
            auto & stream = get_stream_interface(s);
            if(stream.is_enabled()) return static_cast<uint32_t>(stream.get_framerate());
        }
        return 30; // If no streams have yet been enabled, return the minimum possible left/right framerate, to allow the maximum possible exposure range
    }

    /*void r200_camera::set_xu_option(rs_option option, int value)
    {
        if(is_capturing())
        {
            switch(option)
            {
            case RS_OPTION_R200_DEPTH_UNITS:
            case RS_OPTION_R200_DEPTH_CLAMP_MIN:
            case RS_OPTION_R200_DEPTH_CLAMP_MAX:
            case RS_OPTION_R200_DISPARITY_MULTIPLIER:
            case RS_OPTION_R200_DISPARITY_SHIFT:
                throw std::runtime_error("cannot set this option after rs_start_capture(...)");
            }
        }
    }*/
    
    bool r200_camera::supports_option(rs_option option) const
    {
        // We have special logic to implement LR gain and exposure, so they do not belong to the standard option list
        return option == RS_OPTION_R200_LR_GAIN || option == RS_OPTION_R200_LR_EXPOSURE || rs_device::supports_option(option);
    }

    void r200_camera::get_option_range(rs_option option, double & min, double & max, double & step)
    {
        // Gain min/max is framerate dependent
        if(option == RS_OPTION_R200_LR_GAIN)
        {
            r200::set_lr_gain_discovery(get_device(), {get_lr_framerate()});
            auto disc = r200::get_lr_gain_discovery(get_device());
            min = disc.min;
            max = disc.max;
            step = 1;
            return;
        }

        // Exposure min/max is framerate dependent
        if(option == RS_OPTION_R200_LR_EXPOSURE)
        {
            r200::set_lr_exposure_discovery(get_device(), {get_lr_framerate()});
            auto disc = r200::get_lr_exposure_discovery(get_device());
            min = disc.min;
            max = disc.max;
            step = 1;
            return;
        }

        // Default to parent implementation
        rs_device::get_option_range(option, min, max, step);
    }

    // All R200 images which are not in YUY2 format contain an extra row of pixels, called the "dinghy", which contains useful information
    const r200::Dinghy & get_dinghy(const subdevice_mode & mode, const void * frame)
    {
        return *reinterpret_cast<const r200::Dinghy *>(reinterpret_cast<const uint8_t *>(frame) + mode.pf.get_image_size(mode.native_dims.x, mode.native_dims.y-1));
    }

    class dinghy_timestamp_reader : public frame_timestamp_reader
    {
        int max_fps;
    public:
        dinghy_timestamp_reader(int max_fps) : max_fps(max_fps) {}

        bool validate_frame(const subdevice_mode & mode, const void * frame) const override 
        { 
            // No dinghy available on YUY2 images
            if(mode.pf.fourcc == pf_yuy2.fourcc) return true;

            // Check magic number for all subdevices
            auto & dinghy = get_dinghy(mode, frame);
            const uint32_t magic_numbers[] = {0x08070605, 0x04030201, 0x8A8B8C8D};
            if(dinghy.magicNumber != magic_numbers[mode.subdevice])
            {
                LOG_WARNING("Subdevice " << mode.subdevice << " bad magic number 0x" << std::hex << dinghy.magicNumber);
                return false;
            }

            // Check frame status for left/right/Z subdevices only
            if(dinghy.frameStatus != 0 && mode.subdevice != 2)
            {
                LOG_WARNING("Subdevice " << mode.subdevice << " frame status 0x" << std::hex << dinghy.frameStatus);
                return false;
            }

            // Check VDF error status for all subdevices
            if(dinghy.VDFerrorStatus != 0)
            {
                LOG_WARNING("Subdevice " << mode.subdevice << " VDF error status 0x" << std::hex << dinghy.VDFerrorStatus);
                return false;
            }

            // Check CAM module status for left/right subdevice only
            if (dinghy.CAMmoduleStatus != 0 && mode.subdevice == 0)
            {
                LOG_WARNING("Subdevice " << mode.subdevice << " CAM module status 0x" << std::hex << dinghy.CAMmoduleStatus);
                return false;
            }        
            
            // TODO: Check for missing or duplicate frame numbers
            return true;
        }

        int get_frame_timestamp(const subdevice_mode & mode, const void * frame) override 
        { 
            int frame_number = 0;
            if(mode.pf.fourcc == pf_yuy2.fourcc)
            {
                // YUY2 images encode the frame number in the low order bits of the final 32 bytes of the image
                auto data = reinterpret_cast<const uint8_t *>(frame) + ((mode.native_dims.x * mode.native_dims.y) - 32) * 2;
                for(int i = 0; i < 32; ++i)
                {
                    frame_number |= ((*data & 1) << (i & 1 ? 32 - i : 30 - i));
                    data += 2;
                }
            }
            else frame_number = get_dinghy(mode, frame).frameCount; // All other formats can use the frame number in the dinghy row
            return frame_number * 1000 / max_fps;
        }
    };

    class serial_timestamp_generator : public frame_timestamp_reader
    {
        int fps, serial_frame_number;
    public:
        serial_timestamp_generator(int fps) : fps(fps), serial_frame_number() {}

        bool validate_frame(const subdevice_mode & mode, const void * frame) const override { return true; }
        int get_frame_timestamp(const subdevice_mode &, const void *) override 
        { 
            ++serial_frame_number;
            return serial_frame_number * 1000 / fps;
        }
    };

    std::shared_ptr<frame_timestamp_reader> r200_camera::create_frame_timestamp_reader() const
    {
        // If left, right, or Z streams are enabled, convert frame numbers to millisecond timestamps based on LRZ framerate
        for(auto s : {RS_STREAM_DEPTH, RS_STREAM_INFRARED, RS_STREAM_INFRARED2})
        {
            auto & si = get_stream_interface(s);
            if(si.is_enabled()) return std::make_shared<dinghy_timestamp_reader>(si.get_framerate());
        }

        // If only color stream is enabled, generate serial frame timestamps (no HW frame numbers available)
        auto & si = get_stream_interface(RS_STREAM_COLOR);
        if(si.is_enabled()) return std::make_shared<serial_timestamp_generator>(si.get_framerate());

        // No streams enabled, so no need for a timestamp converter
        return nullptr;
    }
}
