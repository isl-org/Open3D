// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "f200.h"
#include "f200-private.h"
#include "image.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>
#include <climits>
#include <algorithm>

namespace rsimpl
{
    static rs_intrinsics MakeDepthIntrinsics(const f200::CameraCalibrationParameters & c, const int2 & dims)
    {
        return {dims.x, dims.y, (c.Kc[0][2]*0.5f + 0.5f) * dims.x, (c.Kc[1][2]*0.5f + 0.5f) * dims.y, c.Kc[0][0]*0.5f * dims.x, c.Kc[1][1]*0.5f * dims.y,
            RS_DISTORTION_INVERSE_BROWN_CONRADY, {c.Invdistc[0], c.Invdistc[1], c.Invdistc[2], c.Invdistc[3], c.Invdistc[4]}};
    }

    static rs_intrinsics MakeColorIntrinsics(const f200::CameraCalibrationParameters & c, const int2 & dims)
    {
        rs_intrinsics intrin = {dims.x, dims.y, c.Kt[0][2]*0.5f + 0.5f, c.Kt[1][2]*0.5f + 0.5f, c.Kt[0][0]*0.5f, c.Kt[1][1]*0.5f, RS_DISTORTION_NONE};
        if(dims.x*3 == dims.y*4) // If using a 4:3 aspect ratio, adjust intrinsics (defaults to 16:9)
        {
            intrin.fx *= 4.0f/3;
            intrin.ppx *= 4.0f/3;
            intrin.ppx -= 1.0f/6;
        }
        intrin.fx *= dims.x;
        intrin.fy *= dims.y;
        intrin.ppx *= dims.x;
        intrin.ppy *= dims.y;
        return intrin;
    }

    struct f200_mode { int2 dims; std::vector<int> fps; };
    static const f200_mode f200_color_modes[] = {
        {{1920, 1080}, {2,5,15,30}},
        {{1280,  720}, {2,5,15,30}},
        {{ 960,  540}, {2,5,15,30,60}},
        {{ 848,  480}, {2,5,15,30,60}},
        {{ 640,  480}, {2,5,15,30,60}},
        {{ 640,  360}, {2,5,15,30,60}},
        {{ 424,  240}, {2,5,15,30,60}},
        {{ 320,  240}, {2,5,15,30,60}},
        {{ 320,  180}, {2,5,15,30,60}}
    };
    static const f200_mode f200_depth_modes[] = {
        {{640, 480}, {2,5,15,30,60}}, 
        {{640, 240}, {2,5,15,30,60,110}}
    };
    static const f200_mode f200_ir_only_modes[] = {
        {{640, 480}, {30,60,120,240,300}}, 
        {{640, 240}, {30,60,120,240,300}}        
    };
    
    static static_device_info get_f200_info(const f200::CameraCalibrationParameters & c)
    {
        LOG_INFO("Connecting to Intel RealSense F200");

        static_device_info info;
        info.name = {"Intel RealSense F200"};

        // Color modes on subdevice 0
        info.stream_subdevices[RS_STREAM_COLOR] = 0;
        for(auto & m : f200_color_modes)
        {
            for(auto fps : m.fps)
            {
                info.subdevice_modes.push_back({0, m.dims, pf_yuy2, fps, MakeColorIntrinsics(c, m.dims), {}, {0}});
            }
        }

        // Depth and IR modes on subdevice 1
        info.stream_subdevices[RS_STREAM_DEPTH] = 1;
        info.stream_subdevices[RS_STREAM_INFRARED] = 1;
        for(auto & m : f200_ir_only_modes)
        {
            for(auto fps : m.fps)
            {
                info.subdevice_modes.push_back({1, m.dims, pf_f200_invi, fps, MakeDepthIntrinsics(c, m.dims), {}, {0}});
            }
        }
        for(auto & m : f200_depth_modes)
        {
            for(auto fps : m.fps)
            {
                info.subdevice_modes.push_back({1, m.dims, pf_invz, fps, MakeDepthIntrinsics(c, m.dims), {}, {0}});       
                info.subdevice_modes.push_back({1, m.dims, pf_f200_inzi, fps, MakeDepthIntrinsics(c, m.dims), {}, {0}});
            }
        }

        info.presets[RS_STREAM_INFRARED][RS_PRESET_BEST_QUALITY] = {true, 640, 480, RS_FORMAT_Y8,   60};
        info.presets[RS_STREAM_DEPTH   ][RS_PRESET_BEST_QUALITY] = {true, 640, 480, RS_FORMAT_Z16,  60};
        info.presets[RS_STREAM_COLOR   ][RS_PRESET_BEST_QUALITY] = {true, 640, 480, RS_FORMAT_RGB8, 60};

        info.presets[RS_STREAM_INFRARED][RS_PRESET_LARGEST_IMAGE] = {true,  640,  480, RS_FORMAT_Y8,   60};
        info.presets[RS_STREAM_DEPTH   ][RS_PRESET_LARGEST_IMAGE] = {true,  640,  480, RS_FORMAT_Z16,  60};
        info.presets[RS_STREAM_COLOR   ][RS_PRESET_LARGEST_IMAGE] = {true, 1920, 1080, RS_FORMAT_RGB8, 60};

        info.presets[RS_STREAM_INFRARED][RS_PRESET_HIGHEST_FRAMERATE] = {true, 640, 480, RS_FORMAT_Y8,   60};
        info.presets[RS_STREAM_DEPTH   ][RS_PRESET_HIGHEST_FRAMERATE] = {true, 640, 480, RS_FORMAT_Z16,  60};
        info.presets[RS_STREAM_COLOR   ][RS_PRESET_HIGHEST_FRAMERATE] = {true, 640, 480, RS_FORMAT_RGB8, 60};

        info.options = {
            {RS_OPTION_F200_LASER_POWER,          0.0, 16.0,  1.0},
            {RS_OPTION_F200_ACCURACY,             1.0, 3.0,   1.0},
            {RS_OPTION_F200_MOTION_RANGE,         0.0, 100.0, 1.0},
            {RS_OPTION_F200_FILTER_OPTION,        0.0, 7.0,   1.0},
            {RS_OPTION_F200_CONFIDENCE_THRESHOLD, 0.0, 15.0,  1.0}
        };

        rsimpl::pose depth_to_color = {transpose((const float3x3 &)c.Rt), (const float3 &)c.Tt * 0.001f}; // convert mm to m
        info.stream_poses[RS_STREAM_DEPTH] = info.stream_poses[RS_STREAM_INFRARED] = inverse(depth_to_color);
        info.stream_poses[RS_STREAM_COLOR] = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}};

        info.nominal_depth_scale = (c.Rmax / 0xFFFF) * 0.001f; // convert mm to m
        info.num_libuvc_transfer_buffers = 1;
        return info;
    }

    static const f200_mode sr300_color_modes[] = {
        {{1920, 1080}, {5,15,30}},
        {{1280,  720}, {5,15,30,60}},
        {{ 960,  540}, {5,15,30,60}},
        {{ 848,  480}, {5,15,30,60}},
        {{ 640,  480}, {5,15,30,60}},
        {{ 640,  360}, {5,15,30,60}},
        {{ 424,  240}, {5,15,30,60}},
        {{ 320,  240}, {5,15,30,60}},
        {{ 320,  180}, {5,15,30,60}}
    };
    static const f200_mode sr300_depth_modes[] = {
        {{640, 480}, {5,15,30,60}}, 
        {{640, 240}, {5,15,30,60,110}}
    };
    static const f200_mode sr300_ir_only_modes[] = {
        {{640, 480}, {30,60,120,200}}      
    };    

    static static_device_info get_sr300_info(const f200::CameraCalibrationParameters & c)
    {
        LOG_INFO("Connecting to Intel RealSense SR300");

        static_device_info info;
        info.name = {"Intel RealSense SR300"};
        
        // Color modes on subdevice 0
        info.stream_subdevices[RS_STREAM_COLOR] = 0;
        for(auto & m : sr300_color_modes)
        {
            for(auto fps : m.fps)
            {
                info.subdevice_modes.push_back({0, m.dims, pf_yuy2, fps, MakeColorIntrinsics(c, m.dims), {}, {0}});
            }
        }

        // Depth and IR modes on subdevice 1
        info.stream_subdevices[RS_STREAM_DEPTH] = 1;
        info.stream_subdevices[RS_STREAM_INFRARED] = 1;
        for(auto & m : sr300_ir_only_modes)
        {
            for(auto fps : m.fps)
            {
                info.subdevice_modes.push_back({1, m.dims, pf_sr300_invi, fps, MakeDepthIntrinsics(c, m.dims), {}, {0}});             
            }
        }
        for(auto & m : sr300_depth_modes)
        {
            for(auto fps : m.fps)
            {
                info.subdevice_modes.push_back({1, m.dims, pf_invz, fps, MakeDepthIntrinsics(c, m.dims), {}, {0}});       
                info.subdevice_modes.push_back({1, m.dims, pf_sr300_inzi, fps, MakeDepthIntrinsics(c, m.dims), {}, {0}});
            }
        }

        for(int i=0; i<RS_PRESET_COUNT; ++i)
        {
            info.presets[RS_STREAM_COLOR   ][i] = {true, 640, 480, RS_FORMAT_RGB8, 60};
            info.presets[RS_STREAM_DEPTH   ][i] = {true, 640, 480, RS_FORMAT_Z16, 60};
            info.presets[RS_STREAM_INFRARED][i] = {true, 640, 480, RS_FORMAT_Y16, 60};
        }

        info.options = {
            {RS_OPTION_F200_LASER_POWER,                            0.0,        16.0,        1.0},
            {RS_OPTION_F200_ACCURACY,                               1.0,        3.0,         1.0},
            {RS_OPTION_F200_MOTION_RANGE,                           0.0,        100.0,       1.0},
            {RS_OPTION_F200_FILTER_OPTION,                          0.0,        7.0,         1.0},
            {RS_OPTION_F200_CONFIDENCE_THRESHOLD,                   0.0,        15.0,        1.0},

            {RS_OPTION_SR300_DYNAMIC_FPS,                           0.0f,       0.0,         0.0}, //2.0,        60.0,        1.0},
            {RS_OPTION_SR300_AUTO_RANGE_ENABLE_MOTION_VERSUS_RANGE, 0.0,        2.0,         1.0},
            {RS_OPTION_SR300_AUTO_RANGE_ENABLE_LASER,               0.0,        1.0,         1.0},  
            {RS_OPTION_SR300_AUTO_RANGE_MIN_MOTION_VERSUS_RANGE,    (double)SHRT_MIN, (double)SHRT_MAX,  1.0}, 
            {RS_OPTION_SR300_AUTO_RANGE_MAX_MOTION_VERSUS_RANGE,    (double)SHRT_MIN, (double)SHRT_MAX,  1.0}, 
            {RS_OPTION_SR300_AUTO_RANGE_START_MOTION_VERSUS_RANGE,  (double)SHRT_MIN, (double)SHRT_MAX,  1.0}, 
            {RS_OPTION_SR300_AUTO_RANGE_MIN_LASER,                  (double)SHRT_MIN, (double)SHRT_MAX,  1.0}, 
            {RS_OPTION_SR300_AUTO_RANGE_MAX_LASER,                  (double)SHRT_MIN, (double)SHRT_MAX,  1.0}, 
            {RS_OPTION_SR300_AUTO_RANGE_START_LASER,                (double)SHRT_MIN, (double)SHRT_MAX,  1.0}, 
            {RS_OPTION_SR300_AUTO_RANGE_UPPER_THRESHOLD,            0.0,        (double)USHRT_MAX,       1.0},
            {RS_OPTION_SR300_AUTO_RANGE_LOWER_THRESHOLD,            0.0,        (double)USHRT_MAX,       1.0},
            {RS_OPTION_SR300_WAKEUP_DEV_PHASE1_PERIOD,              0.0,        (double)USHRT_MAX,       1.0 },
            {RS_OPTION_SR300_WAKEUP_DEV_PHASE1_FPS,                 0.0,        ((double)sr300::e_suspend_fps::eFPS_MAX) - 1,   1.0 },
            {RS_OPTION_SR300_WAKEUP_DEV_PHASE2_PERIOD,              0.0,        (double)USHRT_MAX,                  1.0 },
            {RS_OPTION_SR300_WAKEUP_DEV_PHASE2_FPS,                 0.0,        ((double)sr300::e_suspend_fps::eFPS_MAX) - 1,   1.0 },
            {RS_OPTION_SR300_WAKEUP_DEV_RESET,                      0.0,        0.0,    1.0 },
            {RS_OPTION_SR300_WAKE_ON_USB_REASON,                    0.0,        (double)sr300::wakeonusb_reason::eMaxWakeOnReason, 1.0 },
            {RS_OPTION_SR300_WAKE_ON_USB_CONFIDENCE,                0.0,        100.,   1.0 }  // Percentage
        };

        rsimpl::pose depth_to_color = {transpose((const float3x3 &)c.Rt), (const float3 &)c.Tt * 0.001f}; // convert mm to m
        info.stream_poses[RS_STREAM_DEPTH] = info.stream_poses[RS_STREAM_INFRARED] = inverse(depth_to_color);
        info.stream_poses[RS_STREAM_COLOR] = {{{1,0,0},{0,1,0},{0,0,1}}, {0,0,0}};

        info.nominal_depth_scale = (c.Rmax / 0xFFFF) * 0.001f; // convert mm to m
        info.num_libuvc_transfer_buffers = 1;
        return info;
    }

    f200_camera::f200_camera(std::shared_ptr<uvc::device> device, const static_device_info & info, const f200::CameraCalibrationParameters & calib, const f200::IVCAMTemperatureData & temp, const f200::IVCAMThermalLoopParams & params) :
        rs_device(device, info), base_calibration(calib), base_temperature_data(temp), thermal_loop_params(params), last_temperature_delta(std::numeric_limits<float>::infinity())
    {
        // If thermal control loop requested, start up thread to handle it
		if(thermal_loop_params.IRThermalLoopEnable)
        {
            runTemperatureThread = true;
            temperatureThread = std::thread(&f200_camera::temperature_control_loop, this);
        }

        // These settings come from the "Common" preset. There is no actual way to read the current values off the device.
        arr.enableMvR = 1;
        arr.enableLaser = 1;
        arr.minMvR = 180;
        arr.maxMvR = 605;
        arr.startMvR = 303;
        arr.minLaser = 2;
        arr.maxLaser = 16;
        arr.startLaser = -1;
        arr.ARUpperTh = 1250;
        arr.ARLowerTh = 650;
    }

    //const uvc::guid IVCAM_COLOR_XU = {0xB8EC416E,0xA3AC,0x4580,{0x8D,0x5C,0x0B,0xEE,0x15,0x97,0xE4,0x3D}};

    std::shared_ptr<rs_device> make_f200_device(std::shared_ptr<uvc::device> device)
    {
        std::timed_mutex mutex;
        f200::claim_ivcam_interface(*device);
        auto calib = f200::read_f200_calibration(*device, mutex);
        f200::enable_timestamp(*device, mutex, true, true);

        auto info = get_f200_info(std::get<0>(calib));
        f200::get_module_serial_string(*device, mutex, info.serial, 96);
        f200::get_firmware_version_string(*device, mutex, info.firmware_version);

        return std::make_shared<f200_camera>(device, info, std::get<0>(calib), std::get<1>(calib), std::get<2>(calib));
    }

    std::shared_ptr<rs_device> make_sr300_device(std::shared_ptr<uvc::device> device)
    {
        std::timed_mutex mutex;
        f200::claim_ivcam_interface(*device);
        auto calib = f200::read_sr300_calibration(*device, mutex);
        f200::enable_timestamp(*device, mutex, true, true);

        uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_BACKLIGHT_COMPENSATION, 0);
        uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_BRIGHTNESS, 0);
        uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_CONTRAST, 50);
        uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_GAMMA, 300);
        uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_HUE, 0);
        uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_SATURATION, 64);
        uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_SHARPNESS, 50);
        uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_GAIN, 64);
        //uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_WHITE_BALANCE, 4600); // auto
        //uvc::set_pu_control_with_retry(*device, 0, rs_option::RS_OPTION_COLOR_EXPOSURE, -6); // auto

        auto info = get_sr300_info(std::get<0>(calib));

        f200::get_module_serial_string(*device, mutex, info.serial, 132);
        f200::get_firmware_version_string(*device, mutex, info.firmware_version);

        return std::make_shared<f200_camera>(device, info, std::get<0>(calib), std::get<1>(calib), std::get<2>(calib));
    }

    f200_camera::~f200_camera()
    {
        // Shut down thermal control loop thread
        runTemperatureThread = false;
        temperatureCv.notify_one();
        if (temperatureThread.joinable())
            temperatureThread.join();        
    }

    void f200_camera::on_before_start(const std::vector<subdevice_mode_selection> & selected_modes)
    {

    }
    
    rs_stream f200_camera::select_key_stream(const std::vector<rsimpl::subdevice_mode_selection> & selected_modes)
    {
        int fps[RS_STREAM_NATIVE_COUNT] = {}, max_fps = 0;
        for(const auto & m : selected_modes)
        {
            for(const auto & output : m.get_outputs())
            {
                fps[output.first] = m.mode.fps;
                max_fps = std::max(max_fps, m.mode.fps);
            }
        }

        // Prefer to sync on depth or infrared, but select the stream running at the fastest framerate
        for (auto s : { RS_STREAM_DEPTH, RS_STREAM_INFRARED2, RS_STREAM_INFRARED, RS_STREAM_COLOR })
        {
            if(fps[s] == max_fps) return s;
        }
        return RS_STREAM_DEPTH;
    }

    void f200_camera::temperature_control_loop()
    {
        const float FcxSlope = base_calibration.Kc[0][0] * thermal_loop_params.FcxSlopeA + thermal_loop_params.FcxSlopeB;
        const float UxSlope = base_calibration.Kc[0][2] * thermal_loop_params.UxSlopeA + base_calibration.Kc[0][0] * thermal_loop_params.UxSlopeB + thermal_loop_params.UxSlopeC;
        const float tempFromHFOV = (tan(thermal_loop_params.HFOVsensitivity*(float)M_PI/360)*(1 + base_calibration.Kc[0][0]*base_calibration.Kc[0][0]))/(FcxSlope * (1 + base_calibration.Kc[0][0] * tan(thermal_loop_params.HFOVsensitivity * (float)M_PI/360)));
        float TempThreshold = thermal_loop_params.TempThreshold; //celcius degrees, the temperatures delta that above should be fixed;
        if (TempThreshold <= 0) TempThreshold = tempFromHFOV;
        if (TempThreshold > tempFromHFOV) TempThreshold = tempFromHFOV;

        std::unique_lock<std::mutex> lock(temperatureMutex);
        while (runTemperatureThread) 
        {
            temperatureCv.wait_for(lock, std::chrono::seconds(10));

            // todo - this will throw if bad, but might periodically fail anyway. try/catch
            try
            {
                float IRTemp = (float)f200::read_ir_temp(get_device(), usbMutex);
                float LiguriaTemp = f200::read_mems_temp(get_device(), usbMutex);

                double IrBaseTemperature = base_temperature_data.IRTemp; //should be taken from the parameters
                double liguriaBaseTemperature = base_temperature_data.LiguriaTemp; //should be taken from the parameters

                // calculate deltas from the calibration and last fix
                double IrTempDelta = IRTemp - IrBaseTemperature;
                double liguriaTempDelta = LiguriaTemp - liguriaBaseTemperature;
                double weightedTempDelta = liguriaTempDelta * thermal_loop_params.LiguriaTempWeight + IrTempDelta * thermal_loop_params.IrTempWeight;
                double tempDeltaFromLastFix = fabs(weightedTempDelta - last_temperature_delta);

                // read intrinsic from the calibration working point
                double Kc11 = base_calibration.Kc[0][0];
                double Kc13 = base_calibration.Kc[0][2];

                // Apply model
                if (tempDeltaFromLastFix >= TempThreshold)
                {
                    // if we are during a transition, fix for after the transition
                    double tempDeltaToUse = weightedTempDelta;
                    if (tempDeltaToUse > 0 && tempDeltaToUse < thermal_loop_params.TransitionTemp)
                    {
                        tempDeltaToUse = thermal_loop_params.TransitionTemp;
                    }

                    // calculate fixed values
                    double fixed_Kc11 = Kc11 + (FcxSlope * tempDeltaToUse) + thermal_loop_params.FcxOffset;
                    double fixed_Kc13 = Kc13 + (UxSlope * tempDeltaToUse) + thermal_loop_params.UxOffset;

                    // write back to intrinsic hfov and vfov
                    auto compensated_calibration = base_calibration;
                    compensated_calibration.Kc[0][0] = (float) fixed_Kc11;
                    compensated_calibration.Kc[1][1] = base_calibration.Kc[1][1] * (float)(fixed_Kc11/Kc11);
                    compensated_calibration.Kc[0][2] = (float) fixed_Kc13;

                    // todo - Pass the current resolution into update_asic_coefficients
                    LOG_INFO("updating asic with new temperature calibration coefficients");
                    update_asic_coefficients(get_device(), usbMutex, compensated_calibration);
                    last_temperature_delta = (float)weightedTempDelta;
                }
            }
            catch(const std::exception & e) { LOG_ERROR("TemperatureControlLoop: " << e.what()); }
        }
    }

    void f200_camera::set_options(const rs_option options[], int count, const double values[])
    {
        auto arr_writer = make_struct_interface<f200::IVCAMAutoRangeRequest>([this]() { return arr; }, [this](f200::IVCAMAutoRangeRequest r) {
            f200::set_auto_range(get_device(), usbMutex, r.enableMvR, r.minMvR, r.maxMvR, r.startMvR, r.enableLaser, r.minLaser, r.maxLaser, r.startLaser, r.ARUpperTh, r.ARLowerTh);
            arr = r;
        });

        auto arr_wakeup_dev_writer = make_struct_interface<sr300::wakeup_dev_params>([this]() { return arr_wakeup_dev_param; }, [this](sr300::wakeup_dev_params param) {
            sr300::set_wakeup_device(get_device(), usbMutex, param.phase1Period, (uint32_t)param.phase1FPS, param.phase2Period, (uint32_t)param.phase2FPS);
            arr_wakeup_dev_param = param;
        });

        for(int i=0; i<count; ++i)
        {
            if(uvc::is_pu_control(options[i]))
            {
                uvc::set_pu_control_with_retry(get_device(), 0, options[i], static_cast<int>(values[i]));
                continue;
            }

            switch(options[i])
            {
            case RS_OPTION_F200_LASER_POWER:          f200::set_laser_power(get_device(), static_cast<uint8_t>(values[i])); break;
            case RS_OPTION_F200_ACCURACY:             f200::set_accuracy(get_device(), static_cast<uint8_t>(values[i])); break;
            case RS_OPTION_F200_MOTION_RANGE:         f200::set_motion_range(get_device(), static_cast<uint8_t>(values[i])); break;
            case RS_OPTION_F200_FILTER_OPTION:        f200::set_filter_option(get_device(), static_cast<uint8_t>(values[i])); break;
            case RS_OPTION_F200_CONFIDENCE_THRESHOLD: f200::set_confidence_threshold(get_device(), static_cast<uint8_t>(values[i])); break;
            case RS_OPTION_SR300_DYNAMIC_FPS:         f200::set_dynamic_fps(get_device(), static_cast<uint8_t>(values[i])); break; // IVCAM 1.5 Only

            case RS_OPTION_SR300_WAKEUP_DEV_RESET:    sr300::reset_wakeup_device(get_device(), usbMutex); break;

            case RS_OPTION_SR300_AUTO_RANGE_ENABLE_MOTION_VERSUS_RANGE: arr_writer.set(&f200::IVCAMAutoRangeRequest::enableMvR, values[i]); break; 
            case RS_OPTION_SR300_AUTO_RANGE_ENABLE_LASER:               arr_writer.set(&f200::IVCAMAutoRangeRequest::enableLaser, values[i]); break;
            case RS_OPTION_SR300_AUTO_RANGE_MIN_MOTION_VERSUS_RANGE:    arr_writer.set(&f200::IVCAMAutoRangeRequest::minMvR, values[i]); break;
            case RS_OPTION_SR300_AUTO_RANGE_MAX_MOTION_VERSUS_RANGE:    arr_writer.set(&f200::IVCAMAutoRangeRequest::maxMvR, values[i]); break;
            case RS_OPTION_SR300_AUTO_RANGE_START_MOTION_VERSUS_RANGE:  arr_writer.set(&f200::IVCAMAutoRangeRequest::startMvR, values[i]); break;
            case RS_OPTION_SR300_AUTO_RANGE_MIN_LASER:                  arr_writer.set(&f200::IVCAMAutoRangeRequest::minLaser, values[i]); break;
            case RS_OPTION_SR300_AUTO_RANGE_MAX_LASER:                  arr_writer.set(&f200::IVCAMAutoRangeRequest::maxLaser, values[i]); break;
            case RS_OPTION_SR300_AUTO_RANGE_START_LASER:                arr_writer.set(&f200::IVCAMAutoRangeRequest::startLaser, values[i]); break;
            case RS_OPTION_SR300_AUTO_RANGE_UPPER_THRESHOLD:            arr_writer.set(&f200::IVCAMAutoRangeRequest::ARUpperTh, values[i]); break;
            case RS_OPTION_SR300_AUTO_RANGE_LOWER_THRESHOLD:            arr_writer.set(&f200::IVCAMAutoRangeRequest::ARLowerTh, values[i]); break;
    
            case RS_OPTION_SR300_WAKEUP_DEV_PHASE1_PERIOD:              arr_wakeup_dev_writer.set(&sr300::wakeup_dev_params::phase1Period, values[i]); break;
            case RS_OPTION_SR300_WAKEUP_DEV_PHASE1_FPS:                 arr_wakeup_dev_writer.set(&sr300::wakeup_dev_params::phase1FPS, (int)values[i]); break;
            case RS_OPTION_SR300_WAKEUP_DEV_PHASE2_PERIOD:              arr_wakeup_dev_writer.set(&sr300::wakeup_dev_params::phase2Period, values[i]); break;
            case RS_OPTION_SR300_WAKEUP_DEV_PHASE2_FPS:                 arr_wakeup_dev_writer.set(&sr300::wakeup_dev_params::phase2FPS, (int)values[i]); break;

            default: LOG_WARNING("Cannot set " << options[i] << " to " << values[i] << " on " << get_name()); break;
            }
        }

        arr_writer.commit();
        arr_wakeup_dev_writer.commit();
    }

    void f200_camera::get_options(const rs_option options[], int count, double values[])
    {
        auto arr_reader = make_struct_interface<f200::IVCAMAutoRangeRequest>([this]() { return arr; }, [this](f200::IVCAMAutoRangeRequest r) {});

        for(int i=0; i<count; ++i)
        {
            LOG_INFO("Reading option " << options[i]);

            if(uvc::is_pu_control(options[i]))
            {
                values[i] = uvc::get_pu_control(get_device(), 0, options[i]);
                continue;
            }

            uint8_t val=0;
            switch(options[i])
            {
            case RS_OPTION_F200_LASER_POWER:          f200::get_laser_power         (get_device(), val); values[i] = val; break;
            case RS_OPTION_F200_ACCURACY:             f200::get_accuracy            (get_device(), val); values[i] = val; break;
            case RS_OPTION_F200_MOTION_RANGE:         f200::get_motion_range        (get_device(), val); values[i] = val; break;
            case RS_OPTION_F200_FILTER_OPTION:        f200::get_filter_option       (get_device(), val); values[i] = val; break;
            case RS_OPTION_F200_CONFIDENCE_THRESHOLD: f200::get_confidence_threshold(get_device(), val); values[i] = val; break;
            case RS_OPTION_SR300_DYNAMIC_FPS:         f200::get_dynamic_fps         (get_device(), val); values[i] = val; break; // IVCAM 1.5 Only

            case RS_OPTION_SR300_AUTO_RANGE_ENABLE_MOTION_VERSUS_RANGE: values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::enableMvR); break; 
            case RS_OPTION_SR300_AUTO_RANGE_ENABLE_LASER:               values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::enableLaser); break;
            case RS_OPTION_SR300_AUTO_RANGE_MIN_MOTION_VERSUS_RANGE:    values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::minMvR); break;
            case RS_OPTION_SR300_AUTO_RANGE_MAX_MOTION_VERSUS_RANGE:    values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::maxMvR); break;
            case RS_OPTION_SR300_AUTO_RANGE_START_MOTION_VERSUS_RANGE:  values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::startMvR); break;
            case RS_OPTION_SR300_AUTO_RANGE_MIN_LASER:                  values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::minLaser); break;
            case RS_OPTION_SR300_AUTO_RANGE_MAX_LASER:                  values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::maxLaser); break;
            case RS_OPTION_SR300_AUTO_RANGE_START_LASER:                values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::startLaser); break;
            case RS_OPTION_SR300_AUTO_RANGE_UPPER_THRESHOLD:            values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::ARUpperTh); break;
            case RS_OPTION_SR300_AUTO_RANGE_LOWER_THRESHOLD:            values[i] = arr_reader.get(&f200::IVCAMAutoRangeRequest::ARLowerTh); break;

            case RS_OPTION_SR300_WAKE_ON_USB_REASON:        sr300::get_wakeup_reason(get_device(), usbMutex, val); values[i] = val; break;
            case RS_OPTION_SR300_WAKE_ON_USB_CONFIDENCE:    sr300::get_wakeup_confidence(get_device(), usbMutex, val); values[i] = val; break;

            default: LOG_WARNING("Cannot get " << options[i] << " on " << get_name()); break;
            }
        }
    }

    // TODO: This may need to be modified for thread safety
    class rolling_timestamp_reader : public frame_timestamp_reader
    {
        bool started;
        int64_t total;
        int last_timestamp;
    public:
        rolling_timestamp_reader() : started(), total() {}
        
        bool validate_frame(const subdevice_mode & mode, const void * frame) const override
        { 
            // Validate that at least one byte of the image is nonzero
            for(const uint8_t * it = (const uint8_t *)frame, * end = it + mode.pf.get_image_size(mode.native_dims.x, mode.native_dims.y); it != end; ++it)
            {
                if(*it)
                {
                    return true;
                }
            }

            // F200 and SR300 can sometimes produce empty frames shortly after starting, ignore them
            LOG_INFO("Subdevice " << mode.subdevice << " produced empty frame");
            return false;
        }

        int get_frame_timestamp(const subdevice_mode & mode, const void * frame) override 
        {
            // Timestamps are encoded within the first 32 bits of the image
            int rolling_timestamp =  *reinterpret_cast<const int32_t *>(frame);

            if(!started)
            {
                last_timestamp = rolling_timestamp;
                started = true;
            }

            const int delta = rolling_timestamp - last_timestamp; // NOTE: Relies on undefined behavior: signed int wraparound
            last_timestamp = rolling_timestamp;
            total += delta;
            const int timestamp = static_cast<int>(total / 100000);
            return timestamp;
        }
    };

    std::shared_ptr<frame_timestamp_reader> f200_camera::create_frame_timestamp_reader() const
    {
        return std::make_shared<rolling_timestamp_reader>();
    }

} // namespace rsimpl::f200
