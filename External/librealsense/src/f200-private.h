// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once
#ifndef LIBREALSENSE_F200_PRIVATE_H
#define LIBREALSENSE_F200_PRIVATE_H

#include "uvc.h"
#include <mutex>

namespace rsimpl { namespace f200
{
    struct CameraCalibrationParameters
    {
        float Rmax;
        float Kc[3][3];     // [3x3]: intrinsic calibration matrix of the IR camera
        float Distc[5];     // [1x5]: forward distortion parameters of the IR camera
        float Invdistc[5];  // [1x5]: the inverse distortion parameters of the IR camera
        float Pp[3][4];     // [3x4]: projection matrix
        float Kp[3][3];     // [3x3]: intrinsic calibration matrix of the projector
        float Rp[3][3];     // [3x3]: extrinsic calibration matrix of the projector
        float Tp[3];        // [1x3]: translation vector of the projector
        float Distp[5];     // [1x5]: forward distortion parameters of the projector
        float Invdistp[5];  // [1x5]: inverse distortion parameters of the projector
        float Pt[3][4];     // [3x4]: IR to RGB (texture mapping) image transformation matrix
        float Kt[3][3];
        float Rt[3][3];
        float Tt[3];
        float Distt[5];     // [1x5]: The inverse distortion parameters of the RGB camera
        float Invdistt[5];
        float QV[6];
    };

    struct IVCAMThermalLoopParams
    {
        float IRThermalLoopEnable = 1;      // enable the mechanism
        float TimeOutA = 10000;             // default time out
        float TimeOutB = 0;                 // reserved
        float TimeOutC = 0;                 // reserved
        float TransitionTemp = 3;           // celcius degrees, the transition temperatures to ignore and use offset;
        float TempThreshold = 2;            // celcius degrees, the temperatures delta that above should be fixed;
        float HFOVsensitivity = 0.025f;
        float FcxSlopeA = -0.003696988f;    // the temperature model fc slope a from slope_hfcx = ref_fcx*a + b
        float FcxSlopeB = 0.005809239f;     // the temperature model fc slope b from slope_hfcx = ref_fcx*a + b
        float FcxSlopeC = 0;                // reserved
        float FcxOffset = 0;                // the temperature model fc offset
        float UxSlopeA = -0.000210918f;     // the temperature model ux slope a from slope_ux = ref_ux*a + ref_fcx*b
        float UxSlopeB = 0.000034253955f;   // the temperature model ux slope b from slope_ux = ref_ux*a + ref_fcx*b
        float UxSlopeC = 0;                 // reserved
        float UxOffset = 0;                 // the temperature model ux offset
        float LiguriaTempWeight = 1;        // the liguria temperature weight in the temperature delta calculations
        float IrTempWeight = 0;             // the Ir temperature weight in the temperature delta calculations
        float AmbientTempWeight = 0;        // reserved
        float Param1 = 0;                   // reserved
        float Param2 = 0;                   // reserved
        float Param3 = 0;                   // reserved
        float Param4 = 0;                   // reserved
        float Param5 = 0;                   // reserved
    };

    struct IVCAMTemperatureData
    {
        float LiguriaTemp;
        float IRTemp;
        float AmbientTemp;
    };

    struct IVCAMAutoRangeRequest
    {
        int enableMvR;          // Send as IVCAMCommand::Param1
        int enableLaser;        // Send as IVCAMCommand::Param2
        int16_t minMvR;         // Copy into IVCAMCommand::data
        int16_t maxMvR;         // "
        int16_t startMvR;       // "
        int16_t minLaser;       // "
        int16_t maxLaser;       // "
        int16_t startLaser;     // "
        int16_t ARUpperTh;      // Copy into IVCAMCommand::data if not -1
        int16_t ARLowerTh;      // "
    };

    // Claim USB interface used for device
    void claim_ivcam_interface(uvc::device & device);

    // Read calibration or device state
    std::tuple<CameraCalibrationParameters, IVCAMTemperatureData, IVCAMThermalLoopParams> read_f200_calibration(uvc::device & device, std::timed_mutex & mutex);
    std::tuple<CameraCalibrationParameters, IVCAMTemperatureData, IVCAMThermalLoopParams> read_sr300_calibration(uvc::device & device, std::timed_mutex & mutex);
    float read_mems_temp(uvc::device & device, std::timed_mutex & mutex);
    int read_ir_temp(uvc::device & device, std::timed_mutex & mutex);
    void get_gvd(uvc::device & device, std::timed_mutex & mutex, size_t sz, char * gvd);
    void get_firmware_version_string(uvc::device & device, std::timed_mutex & mutex, std::string & version);
    void get_module_serial_string(uvc::device & device, std::timed_mutex & mutex, std::string & serial, int offset);

    // Modify device state
    void force_hardware_reset(uvc::device & device, std::timed_mutex & mutex);
    void enable_timestamp(uvc::device & device, std::timed_mutex & mutex, bool colorEnable, bool depthEnable);
    void update_asic_coefficients(uvc::device & device, std::timed_mutex & mutex, const CameraCalibrationParameters & compensated_params); // todo - Allow you to specify resolution
    void set_auto_range(uvc::device & device, std::timed_mutex & mutex, int enableMvR, int16_t minMvR, int16_t maxMvR, int16_t startMvR, int enableLaser, int16_t minLaser, int16_t maxLaser, int16_t startLaser, int16_t ARUpperTH, int16_t ARLowerTH);

    // XU read/write
    void get_laser_power(const uvc::device & device, uint8_t & laser_power);
    void set_laser_power(uvc::device & device, uint8_t laser_power);  
    void get_accuracy(const uvc::device & device, uint8_t & accuracy);  
    void set_accuracy(uvc::device & device, uint8_t accuracy);    
    void get_motion_range(const uvc::device & device, uint8_t & motion_range);
    void set_motion_range(uvc::device & device, uint8_t motion_range);
    void get_filter_option(const uvc::device & device, uint8_t & filter_option);
    void set_filter_option(uvc::device & device, uint8_t filter_option);
    void get_confidence_threshold(const uvc::device & device, uint8_t & conf_thresh);
    void set_confidence_threshold(uvc::device & device, uint8_t conf_thresh);
    void get_dynamic_fps(const uvc::device & device, uint8_t & dynamic_fps);
    void set_dynamic_fps(uvc::device & device, uint8_t dynamic_fps);

    #define NUM_OF_CALIBRATION_COEFFS   (64)

} // rsimpl::f200

namespace sr300
{
    enum class wakeonusb_reason : unsigned char
    {
        eNone_provided = 0,    // Wake-up performed, but FW doesn't provide a comprehensive resason
        eUser_input = 1,    // Firmware recognized user input and performed system wake up accordingly
        eUninitialized = 2,    // Querrying the interface before the FW had a chance to perform  ACPI wake on USB
        eMaxWakeOnReason
    };

    enum class e_suspend_fps : uint32_t
    {
        eFPS_2 = 0,
        eFPS_1,
        eFPS_05,
        eFPS_025,
        eFPS_MAX
    };

    struct wakeup_dev_params
    {
        wakeup_dev_params(void) : phase1Period(UINT32_MAX), phase1FPS(e_suspend_fps::eFPS_MAX), phase2Period(UINT32_MAX), phase2FPS(e_suspend_fps::eFPS_MAX){};
        wakeup_dev_params(uint32_t  p1, e_suspend_fps p2, uint32_t  p3, e_suspend_fps p4) :         phase1Period(p1), phase1FPS(p2), phase2Period(p3), phase2FPS(p4){};
        uint32_t        phase1Period;
        e_suspend_fps   phase1FPS;
        uint32_t        phase2Period;
        e_suspend_fps   phase2FPS;
        bool isValid() { return ((phase1FPS < e_suspend_fps::eFPS_MAX) && (phase2FPS < e_suspend_fps::eFPS_MAX) && (phase1Period<UINT32_MAX) && (phase2Period<UINT32_MAX)); };
    };


    // Wakeup device interfaces
    void set_wakeup_device(uvc::device & device, std::timed_mutex & mutex, const uint32_t&phase1Period, const uint32_t& phase1FPS, const uint32_t&phase2Period, const uint32_t& phase2FPS);
    void reset_wakeup_device(uvc::device & device, std::timed_mutex & mutex);
    void get_wakeup_reason(uvc::device & device, std::timed_mutex & mutex, unsigned char &cReason);
    void get_wakeup_confidence(uvc::device & device, std::timed_mutex & mutex, unsigned char &cConfidence);

} // rsimpl::sr300
} // namespace rsimpl

#endif
