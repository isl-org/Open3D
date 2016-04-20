// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "f200-private.h"

#include <cstring>
#include <algorithm>
#include <thread>
#include <cmath>

#define IV_COMMAND_FIRMWARE_UPDATE_MODE     0x01
#define IV_COMMAND_GET_CALIBRATION_DATA     0x02
#define IV_COMMAND_LASER_POWER              0x03
#define IV_COMMAND_DEPTH_ACCURACY           0x04
#define IV_COMMAND_ZUNIT                    0x05
#define IV_COMMAND_LOW_CONFIDENCE_LEVEL     0x06
#define IV_COMMAND_INTENSITY_IMAGE_TYPE     0x07
#define IV_COMMAND_MOTION_VS_RANGE_TRADE    0x08
#define IV_COMMAND_POWER_GEAR               0x09
#define IV_COMMAND_FILTER_OPTION            0x0A
#define IV_COMMAND_VERSION                  0x0B
#define IV_COMMAND_CONFIDENCE_THRESHHOLD    0x0C

#define IVCAM_MONITOR_INTERFACE         0x4
#define IVCAM_VID                       0x8086
#define IVCAM_PID                       0x0A66
#define IVCAM_MONITOR_ENDPOINT_OUT      0x1
#define IVCAM_MONITOR_ENDPOINT_IN       0x81
#define IVCAM_MONITOR_MAGIC_NUMBER      0xcdab
#define IVCAM_MIN_SUPPORTED_VERSION     13
#define IVCAM_MONITOR_MAX_BUFFER_SIZE   1024
#define IVCAM_MONITOR_HEADER_SIZE       (sizeof(uint32_t)*6)
#define IVCAM_MONITOR_MUTEX_TIMEOUT     3000

#define NUM_OF_CALIBRATION_PARAMS       (100)
#define PARAMETERS_BUFFER_SIZE          (50)
#define MAX_SIZE_OF_CALIB_PARAM_BYTES   (800)
#define SIZE_OF_CALIB_PARAM_BYTES       (512)
#define SIZE_OF_CALIB_HEADER_BYTES      (4)
#define HW_MONITOR_COMMAND_SIZE         (1000)
#define HW_MONITOR_BUFFER_SIZE          (1000)
#define NUM_OF_CALIBRATION_COEFFS       (64)

// IVCAM depth XU identifiers
#define IVCAM_DEPTH_LASER_POWER         1
#define IVCAM_DEPTH_ACCURACY            2
#define IVCAM_DEPTH_MOTION_RANGE        3
#define IVCAM_DEPTH_ERROR               4
#define IVCAM_DEPTH_FILTER_OPTION       5
#define IVCAM_DEPTH_CONFIDENCE_THRESH   6
#define IVCAM_DEPTH_DYNAMIC_FPS         7 // Only available on IVCAM 1.5 / SR300

// IVCAM color XU identifiers
#define IVCAM_COLOR_EXPOSURE_PRIORITY   1
#define IVCAM_COLOR_AUTO_FLICKER        2
#define IVCAM_COLOR_ERROR               3
#define IVCAM_COLOR_EXPOSURE_GRANULAR   4

namespace rsimpl { namespace f200
{
    enum class IVCAMMonitorCommand : uint32_t
    {
        UpdateCalib                = 0xBC,
        GetIRTemp                  = 0x52,
        GetMEMSTemp                = 0x0A,
        HWReset                    = 0x28,
        GVD                        = 0x3B,
        BIST                       = 0xFF,
        GoToDFU                    = 0x80,
        GetCalibrationTable        = 0x3D,
        DebugFormat                = 0x0B,
        TimeStampEnable            = 0x0C,
        GetPowerGearState          = 0xFF,
        SetDefaultControls         = 0xA6,
        GetDefaultControls         = 0xA7,
        GetFWLastError             = 0x0E,
        CheckI2cConnect            = 0x4A,
        CheckRGBConnect            = 0x4B,
        CheckDPTConnect            = 0x4C,
		SetAutoRange			   = 0xA6,
		OnSuspendResume			   = 0x91,
		GetWakeReason			   = 0x93,
		GetWakeConfidence		   = 0x92,
		AutoRangeSetParamsforDebug = 0xb3
    };

    struct IVCAMCommand
    {
        IVCAMMonitorCommand cmd;
        int Param1;
        int Param2;
        int Param3;
        int Param4;
        char data[HW_MONITOR_BUFFER_SIZE];
        int sizeOfSendCommandData;
        long TimeOut;
        bool oneDirection;
        char receivedCommandData[HW_MONITOR_BUFFER_SIZE];
        int receivedCommandDataLength;
        char receivedOpcode[4];

        IVCAMCommand(IVCAMMonitorCommand cmd)
        {
            this->cmd = cmd;
            Param1 = 0;
            Param2 = 0;
            Param3 = 0;
            Param4 = 0;
            sizeOfSendCommandData = 0;
            TimeOut = 5000;
            oneDirection = false;
        }
    };

    struct IVCAMCommandDetails
    {
        bool oneDirection;
        char sendCommandData[HW_MONITOR_COMMAND_SIZE];
        int sizeOfSendCommandData;
        long TimeOut;
        char receivedOpcode[4];
        char receivedCommandData[HW_MONITOR_BUFFER_SIZE];
        int receivedCommandDataLength;
    };

    struct IVCAMASICCoefficients
    {
        float CoefValueArray[NUM_OF_CALIBRATION_COEFFS];
    };

    struct OACOffsetData
    {
        int OACOffset1;
        int OACOffset2;
        int OACOffset3;
        int OACOffset4;
    };

    struct IVCAMTesterData
    {
        int16_t TableValidation;
        int16_t TableVarsion;
		IVCAMTemperatureData TemperatureData;
        OACOffsetData OACOffsetData_;
        IVCAMThermalLoopParams ThermalLoopParams;
    };

    struct IVCAMCalibration
    {
        int uniqueNumber; //Should be 0xCAFECAFE in Calibration version 1 or later. In calibration version 0 this is zero.
        int16_t TableValidation;
        int16_t TableVersion;
        CameraCalibrationParameters CalibrationParameters;
    };

    struct SR300RawCalibration
    {
        uint16_t tableVersion;
        uint16_t tableID;
        uint32_t dataSize;
        uint32_t reserved;
        int crc;
        CameraCalibrationParameters CalibrationParameters;
        uint8_t reserved_1[176];
        IVCAMTemperatureData TemperatureData;
        uint8_t reserved21[148];
    };

    enum class IVCAMDataSource : uint32_t
    {
        TakeFromRO = 0,
        TakeFromRW = 1,
        TakeFromRAM = 2
    };

    enum Property
    {
        IVCAM_PROPERTY_COLOR_EXPOSURE                   =   1,
        IVCAM_PROPERTY_COLOR_BRIGHTNESS                 =   2,
        IVCAM_PROPERTY_COLOR_CONTRAST                   =   3,
        IVCAM_PROPERTY_COLOR_SATURATION                 =   4,
        IVCAM_PROPERTY_COLOR_HUE                        =   5,
        IVCAM_PROPERTY_COLOR_GAMMA                      =   6,
        IVCAM_PROPERTY_COLOR_WHITE_BALANCE              =   7,
        IVCAM_PROPERTY_COLOR_SHARPNESS                  =   8,
        IVCAM_PROPERTY_COLOR_BACK_LIGHT_COMPENSATION    =   9,
        IVCAM_PROPERTY_COLOR_GAIN                       =   10,
        IVCAM_PROPERTY_COLOR_POWER_LINE_FREQUENCY       =   11,
        IVCAM_PROPERTY_AUDIO_MIX_LEVEL                  =   12,
        IVCAM_PROPERTY_APERTURE                         =   13,
        IVCAM_PROPERTY_DISTORTION_CORRECTION_I          =   202,
        IVCAM_PROPERTY_DISTORTION_CORRECTION_DPTH       =   203,
        IVCAM_PROPERTY_DEPTH_MIRROR                     =   204,    //0 - not mirrored, 1 - mirrored
        IVCAM_PROPERTY_COLOR_MIRROR                     =   205,
        IVCAM_PROPERTY_COLOR_FIELD_OF_VIEW              =   207,
        IVCAM_PROPERTY_COLOR_SENSOR_RANGE               =   209,
        IVCAM_PROPERTY_COLOR_FOCAL_LENGTH               =   211,
        IVCAM_PROPERTY_COLOR_PRINCIPAL_POINT            =   213,
        IVCAM_PROPERTY_DEPTH_FIELD_OF_VIEW              =   215,
        IVCAM_PROPERTY_DEPTH_UNDISTORTED_FIELD_OF_VIEW  =   223,
        IVCAM_PROPERTY_DEPTH_SENSOR_RANGE               =   217,
        IVCAM_PROPERTY_DEPTH_FOCAL_LENGTH               =   219,
        IVCAM_PROPERTY_DEPTH_UNDISTORTED_FOCAL_LENGTH   =   225,
        IVCAM_PROPERTY_DEPTH_PRINCIPAL_POINT            =   221,
        IVCAM_PROPERTY_MF_DEPTH_LOW_CONFIDENCE_VALUE    =   5000,
        IVCAM_PROPERTY_MF_DEPTH_UNIT                    =   5001,   // in micron
        IVCAM_PROPERTY_MF_CALIBRATION_DATA              =   5003,
        IVCAM_PROPERTY_MF_LASER_POWER                   =   5004,
        IVCAM_PROPERTY_MF_ACCURACY                      =   5005,
        IVCAM_PROPERTY_MF_INTENSITY_IMAGE_TYPE          =   5006,   //0 - (I0 - laser off), 1 - (I1 - Laser on), 2 - (I1-I0), default is I1.
        IVCAM_PROPERTY_MF_MOTION_VS_RANGE_TRADE         =   5007,
        IVCAM_PROPERTY_MF_POWER_GEAR                    =   5008,
        IVCAM_PROPERTY_MF_FILTER_OPTION                 =   5009,
        IVCAM_PROPERTY_MF_VERSION                       =   5010,
        IVCAM_PROPERTY_MF_DEPTH_CONFIDENCE_THRESHOLD    =   5013,
        IVCAM_PROPERTY_ACCELEROMETER_READING            =   3000,   // three values
        IVCAM_PROPERTY_PROJECTION_SERIALIZABLE          =   3003,   
        IVCAM_PROPERTY_CUSTOMIZED                       =   0x04000000,
    };

    enum class FirmwareError : int32_t
    {
        FW_ACTIVE = 0,
        FW_MSAFE_S1_ERR,
        FW_I2C_SAFE_ERR,
        FW_FLASH_SAFE_ERR,
        FW_I2C_CFG_ERR,
        FW_I2C_EV_ERR,
        FW_HUMIDITY_ERR,
        FW_MSAFE_S0_ERR,
        FW_LD_ERR,
        FW_PI_ERR,
        FW_PJCLK_ERR,
        FW_OAC_ERR,
        FW_LIGURIA_TEMPERATURE_ERR,
        FW_CONTINUE_SAFE_ERROR,
        FW_FORZA_HUNG,
        FW_FORZA_CONTINUES_HUNG,
        FW_PJ_EYESAFETY_CHKRHARD,
        FW_MIPI_PCAM_ERR,
        FW_MIPI_TCAM_ERR,
        FW_SYNC_DISABLED,
        FW_MIPI_PCAM_SVR_ERR,
        FW_MIPI_TCAM_SVR_ERR,
        FW_MIPI_PCAM_FRAME_SIZE_ERR,
        FW_MIPI_TCAM_FRAME_SIZE_ERR,
        FW_MIPI_PCAM_FRAME_RESPONSE_ERR,
        FW_MIPI_TCAM_FRAME_RESPONSE_ERR,
        FW_USB_PCAM_THROTTLED_ERR,
        FW_USB_TCAM_THROTTLED_ERR,
        FW_USB_PCAM_QOS_WAR,
        FW_USB_TCAM_QOS_WAR,
        FW_USB_PCAM_OVERFLOW,
        FW_USB_TCAM_OVERFLOW,
        FW_Flash_OEM_SECTOR,
        FW_Flash_CALIBRATION_RW,
        FW_Flash_IR_CALIBRATION,
        FW_Flash_RGB_CALIBRATION,
        FW_Flash_THERMAL_LOOP_CONFIGURATION,
        FW_Flash_REALTEK,
        FW_RGB_ISP_BOOT_FAILED,
        FW_PRIVACY_RGB_OFF,
        FW_PRIVACY_DEPTH_OFF,
        FW_COUNT_ERROR
    };

    //////////////////
    // XU functions //
    //////////////////

    const uvc::extension_unit depth_xu {1, 6, 1, {0xA55751A1,0xF3C5,0x4A5E,{0x8D,0x5A,0x68,0x54,0xB8,0xFA,0x27,0x16}}};

    // N.B. f200 xu_read and xu_write hard code the xu interface to the depth suvdevice. There is only a
    // single *potentially* useful XU on the color device, so let's ignore it for now.
    void xu_read(const uvc::device & device, uint8_t xu_ctrl, void * buffer, uint32_t length)
    {
        uvc::get_control_with_retry(device, depth_xu, static_cast<int>(xu_ctrl), buffer, length);
    }
    
    void xu_write(uvc::device & device, uint8_t xu_ctrl, void * buffer, uint32_t length)
    {
        uvc::set_control_with_retry(device, depth_xu, static_cast<int>(xu_ctrl), buffer, length);
    }
    
    void get_laser_power(const uvc::device & device, uint8_t & laser_power)
    {
        xu_read(device, IVCAM_DEPTH_LASER_POWER, &laser_power, sizeof(laser_power));
    }
    
    void set_laser_power(uvc::device & device, uint8_t laser_power)
    {
        xu_write(device, IVCAM_DEPTH_LASER_POWER, &laser_power, sizeof(laser_power));
    }
    
    void get_accuracy(const uvc::device & device, uint8_t & accuracy)
    {
        xu_read(device, IVCAM_DEPTH_ACCURACY, &accuracy, sizeof(accuracy));
    }
    
    void set_accuracy(uvc::device & device, uint8_t accuracy)
    {
        xu_write(device, IVCAM_DEPTH_ACCURACY, &accuracy, sizeof(accuracy));
    }
    
    void get_motion_range(const uvc::device & device, uint8_t & motion_range)
    {
        xu_read(device, IVCAM_DEPTH_MOTION_RANGE, &motion_range, sizeof(motion_range));
    }
    
    void set_motion_range(uvc::device & device, uint8_t motion_range)
    {
        xu_write(device, IVCAM_DEPTH_MOTION_RANGE, &motion_range, sizeof(motion_range));
    }
    
    void get_filter_option(const uvc::device & device, uint8_t & filter_option)
    {
        xu_read(device, IVCAM_DEPTH_FILTER_OPTION, &filter_option, sizeof(filter_option));
    }
    
    void set_filter_option(uvc::device & device, uint8_t filter_option)
    {
        xu_write(device, IVCAM_DEPTH_FILTER_OPTION, &filter_option, sizeof(filter_option));
    }
    
    void get_confidence_threshold(const uvc::device & device, uint8_t & conf_thresh)
    {
        xu_read(device, IVCAM_DEPTH_CONFIDENCE_THRESH, &conf_thresh, sizeof(conf_thresh));
    }
    
    void set_confidence_threshold(uvc::device & device, uint8_t conf_thresh)
    {
        xu_write(device, IVCAM_DEPTH_CONFIDENCE_THRESH, &conf_thresh, sizeof(conf_thresh));
    }
    
    void get_dynamic_fps(const uvc::device & device, uint8_t & dynamic_fps)
    {
        return xu_read(device, IVCAM_DEPTH_DYNAMIC_FPS, &dynamic_fps, sizeof(dynamic_fps));
    }
    
    void set_dynamic_fps(uvc::device & device, uint8_t dynamic_fps)
    {
        return xu_write(device, IVCAM_DEPTH_DYNAMIC_FPS, &dynamic_fps, sizeof(dynamic_fps));
    }

    ///////////////////
    // USB functions //
    ///////////////////

    void claim_ivcam_interface(uvc::device & device)
    {
        const uvc::guid IVCAM_WIN_USB_DEVICE_GUID = {0x175695CD, 0x30D9, 0x4F87, {0x8B, 0xE3, 0x5A, 0x82, 0x70, 0xF4, 0x9A, 0x31}};
        claim_interface(device, IVCAM_WIN_USB_DEVICE_GUID, IVCAM_MONITOR_INTERFACE);
    }

    int prepare_usb_command(uint8_t * request, size_t & requestSize, uint32_t op, uint32_t p1 = 0, uint32_t p2 = 0, uint32_t p3 = 0, uint32_t p4 = 0, uint8_t * data = 0, size_t dataLength = 0)
    {

        if (requestSize < IVCAM_MONITOR_HEADER_SIZE)
            return 0;

        int index = sizeof(uint16_t);
        *(uint16_t *)(request + index) = IVCAM_MONITOR_MAGIC_NUMBER;
        index += sizeof(uint16_t);
        *(uint32_t *)(request + index) = op;
        index += sizeof(uint32_t);
        *(uint32_t *)(request + index) = p1;
        index += sizeof(uint32_t);
        *(uint32_t *)(request + index) = p2;
        index += sizeof(uint32_t);
        *(uint32_t *)(request + index) = p3;
        index += sizeof(uint32_t);
        *(uint32_t *)(request + index) = p4;
        index += sizeof(uint32_t);

        if (dataLength)
        {
            memcpy(request + index , data, dataLength);
            index += dataLength;
        }

        // Length doesn't include header size (sizeof(uint32_t))
        *(uint16_t *)request = (uint16_t)(index - sizeof(uint32_t));
        requestSize = index;
        return index;
    }

    // NOTE: Almost certainly redundant with above function, unify later
    void fill_usb_buffer(int opCodeNumber, int p1, int p2, int p3, int p4, char * data, int dataLength, char * bufferToSend, int & length)
    {
        uint16_t preHeaderData = IVCAM_MONITOR_MAGIC_NUMBER;

        char * writePtr = bufferToSend;
        int header_size = 4;

        // TBD !!! This may change. Need to define it as part of API
        //typeSize = sizeof(float);

        int cur_index = 2;
        *(uint16_t *)(writePtr + cur_index) = preHeaderData;
        cur_index += sizeof(uint16_t);
        *(int *)( writePtr + cur_index) = opCodeNumber;
        cur_index += sizeof(uint32_t);
        *(int *)( writePtr + cur_index) = p1;
        cur_index += sizeof(uint32_t);
        *(int *)( writePtr + cur_index) = p2;
        cur_index += sizeof(uint32_t);
        *(int *)( writePtr + cur_index) = p3;
        cur_index += sizeof(uint32_t);
        *(int *)( writePtr + cur_index) = p4;
        cur_index += sizeof(uint32_t);

        if (dataLength)
        {
            memcpy(writePtr + cur_index , data, dataLength);
            cur_index += dataLength;
        }

        length = cur_index;
        *(uint16_t *) bufferToSend = (uint16_t)(length - header_size); // Length doesn't include header
    }

    void execute_usb_command(uvc::device & device, std::timed_mutex & mutex, uint8_t *out, size_t outSize, uint32_t & op, uint8_t * in, size_t & inSize)
    {
        // write
        errno = 0;

        int outXfer;

        if (!mutex.try_lock_for(std::chrono::milliseconds(IVCAM_MONITOR_MUTEX_TIMEOUT))) throw std::runtime_error("timed_mutex::try_lock_for(...) timed out");
        std::lock_guard<std::timed_mutex> guard(mutex, std::adopt_lock);

        bulk_transfer(device, IVCAM_MONITOR_ENDPOINT_OUT, out, (int) outSize, &outXfer, 1000); // timeout in ms

        // read
        if (in && inSize)
        {
            uint8_t buf[IVCAM_MONITOR_MAX_BUFFER_SIZE];

            errno = 0;

            bulk_transfer(device, IVCAM_MONITOR_ENDPOINT_IN, buf, sizeof(buf), &outXfer, 1000);
            if (outXfer < (int)sizeof(uint32_t)) throw std::runtime_error("incomplete bulk usb transfer");

            op = *(uint32_t *)buf;
            if (outXfer > (int)inSize) throw std::runtime_error("bulk transfer failed - user buffer too small");
            inSize = outXfer;
            memcpy(in, buf, inSize);
        }
    }

    void send_hw_monitor_command(uvc::device & device, std::timed_mutex & mutex, IVCAMCommandDetails & details)
    {
        unsigned char outputBuffer[HW_MONITOR_BUFFER_SIZE];

        uint32_t op;
        size_t receivedCmdLen = HW_MONITOR_BUFFER_SIZE;

        execute_usb_command(device, mutex, (uint8_t*) details.sendCommandData, (size_t) details.sizeOfSendCommandData, op, outputBuffer, receivedCmdLen);
        details.receivedCommandDataLength = receivedCmdLen;

        if (details.oneDirection) return;

        if (details.receivedCommandDataLength < 4) throw std::runtime_error("received incomplete response to usb command");

        details.receivedCommandDataLength -= 4;
        memcpy(details.receivedOpcode, outputBuffer, 4);

        if (details.receivedCommandDataLength > 0 )
            memcpy(details.receivedCommandData, outputBuffer + 4, details.receivedCommandDataLength);
    }

    void perform_and_send_monitor_command(uvc::device & device, std::timed_mutex & mutex, IVCAMCommand & newCommand)
    {
        uint32_t opCodeXmit = (uint32_t) newCommand.cmd;

        IVCAMCommandDetails details;
        details.oneDirection = newCommand.oneDirection;
        details.TimeOut = newCommand.TimeOut;

        fill_usb_buffer(opCodeXmit,
                        newCommand.Param1,
                        newCommand.Param2,
                        newCommand.Param3,
                        newCommand.Param4,
                        newCommand.data,
                        newCommand.sizeOfSendCommandData,
                        details.sendCommandData,
                        details.sizeOfSendCommandData);

        send_hw_monitor_command(device, mutex, details);

        // Error/exit conditions
        if (newCommand.oneDirection)
            return;

        memcpy(newCommand.receivedOpcode, details.receivedOpcode, 4);
        memcpy(newCommand.receivedCommandData, details.receivedCommandData,details.receivedCommandDataLength);
        newCommand.receivedCommandDataLength = details.receivedCommandDataLength;

        // endian? 
        uint32_t opCodeAsUint32 = pack(details.receivedOpcode[3], details.receivedOpcode[2], details.receivedOpcode[1], details.receivedOpcode[0]);
        if (opCodeAsUint32 != opCodeXmit)
        {
            throw std::runtime_error("opcodes do not match");
        }
    }

    // "Get Version and Date"
    // Reference: Commands.xml in IVCAM_DLL
    void get_gvd(uvc::device & device, std::timed_mutex & mutex, size_t sz, char * gvd)
    {
        IVCAMCommand cmd(IVCAMMonitorCommand::GVD);
        perform_and_send_monitor_command(device, mutex, cmd);
        auto minSize = std::min((int) sz, cmd.receivedCommandDataLength);
        memcpy(gvd, cmd.receivedCommandData, minSize);
    }

    void get_firmware_version_string(uvc::device & device, std::timed_mutex & mutex, std::string & version)
    {
        std::vector<char> gvd(1024);
        get_gvd(device, mutex, 1024, gvd.data());
        char fws[8];
        memcpy(fws, gvd.data(), 8); // offset 0
        version = std::string(std::to_string(fws[3]) + "." + std::to_string(fws[2]) + "." + std::to_string(fws[1]) + "." + std::to_string(fws[0]));
    }

    void get_module_serial_string(uvc::device & device, std::timed_mutex & mutex, std::string & serial, int offset)
    {
        std::vector<char> gvd(1024);
        get_gvd(device, mutex, 1024, gvd.data());
        unsigned char ss[8];
        memcpy(ss, gvd.data() + offset, 8);
        char formattedBuffer[64];
        if (offset == 96)
        {
            sprintf(formattedBuffer, "%02X%02X%02X%02X%02X%02X", ss[0], ss[1], ss[2], ss[3], ss[4], ss[5]);
            serial = std::string(formattedBuffer);
        }
        else if (offset == 132)
        {
            sprintf(formattedBuffer, "%02X%02X%02X%02X%02X%-2X", ss[0], ss[1], ss[2], ss[3], ss[4], ss[5]);
            serial = std::string(formattedBuffer);
        }
    }

    void force_hardware_reset(uvc::device & device, std::timed_mutex & mutex)
    {
        IVCAMCommand cmd(IVCAMMonitorCommand::HWReset);
        cmd.oneDirection = true;
        perform_and_send_monitor_command(device, mutex, cmd);
    }

    void enable_timestamp(uvc::device & device, std::timed_mutex & mutex, bool colorEnable, bool depthEnable)
    {
        IVCAMCommand cmd(IVCAMMonitorCommand::TimeStampEnable);
        cmd.Param1 = depthEnable ? 1 : 0;
        cmd.Param2 = colorEnable ? 1 : 0;
        perform_and_send_monitor_command(device, mutex, cmd);
    }

    void set_asic_coefficients(uvc::device & device, std::timed_mutex & mutex, const IVCAMASICCoefficients & coeffs)
    {
        // command.Param1 =
        // 0 - INVZ VGA (640x480)
        // 1 - INVZ QVGA (Possibly 320x240?)
        // 2 - INVZ HVGA (640x240)
        // 3 - INVZ 640x360
        // 4 - INVR VGA (640x480)
        // 5 - INVR QVGA (Possibly 320x240?)
        // 6 - INVR HVGA (640x240)
        // 7 - INVR 640x360

         IVCAMCommand command(IVCAMMonitorCommand::UpdateCalib);

         memcpy(command.data, coeffs.CoefValueArray, NUM_OF_CALIBRATION_COEFFS * sizeof(float));
         command.Param1 = 0; // todo - Send appropriate value at appropriate times, see above
         command.Param2 = 0;
         command.Param3 = 0;
         command.Param4 = 0;
         command.oneDirection = false;
         command.sizeOfSendCommandData = NUM_OF_CALIBRATION_COEFFS * sizeof(float);
         command.TimeOut = 5000;

         perform_and_send_monitor_command(device, mutex, command);
    }

    void set_auto_range(uvc::device & device, std::timed_mutex & mutex, int enableMvR, int16_t minMvR, int16_t maxMvR, int16_t startMvR, int enableLaser, int16_t minLaser, int16_t maxLaser, int16_t startLaser, int16_t ARUpperTH, int16_t ARLowerTH)
    {
	    IVCAMCommand CommandParameters(IVCAMMonitorCommand::SetAutoRange);
	    CommandParameters.Param1 = enableMvR;
	    CommandParameters.Param2 = enableLaser;

	    auto data = reinterpret_cast<int16_t *>(CommandParameters.data);
	    data[0] = minMvR;
	    data[1] = maxMvR;
	    data[2] = startMvR;
	    data[3] = minLaser;
	    data[4] = maxLaser;
	    data[5] = startLaser;
	    auto size = 12;

	    if (ARUpperTH != -1)
	    {
		    data[6] = ARUpperTH;
		    size += 2;
	    }	

	    if (ARLowerTH != -1)
	    {
		    data[7] = ARLowerTH;
		    size += 2;
	    }
		
	    CommandParameters.sizeOfSendCommandData = size;
        perform_and_send_monitor_command(device, mutex, CommandParameters);
    }

    FirmwareError get_fw_last_error(uvc::device & device, std::timed_mutex & mutex)
    {
        IVCAMCommand cmd(IVCAMMonitorCommand::GetFWLastError);
        memset(cmd.data, 0, 4);

        perform_and_send_monitor_command(device, mutex, cmd);
        return *reinterpret_cast<FirmwareError *>(cmd.receivedCommandData);
    }

    float read_mems_temp(uvc::device & device, std::timed_mutex & mutex)
    {
        IVCAMCommand command(IVCAMMonitorCommand::GetMEMSTemp);
        command.Param1 = 0;
        command.Param2 = 0;
        command.Param3 = 0;
        command.Param4 = 0;
        command.sizeOfSendCommandData = 0;
        command.TimeOut = 5000;
        command.oneDirection = false;

        perform_and_send_monitor_command(device, mutex, command);
        int32_t t = *reinterpret_cast<int32_t *>(command.receivedCommandData);
        return static_cast<float>(t) / 100;
    }

    int read_ir_temp(uvc::device & device, std::timed_mutex & mutex)
    {
        IVCAMCommand command(IVCAMMonitorCommand::GetIRTemp);
        command.Param1 = 0;
        command.Param2 = 0;
        command.Param3 = 0;
        command.Param4 = 0;
        command.sizeOfSendCommandData = 0;
        command.TimeOut = 5000;
        command.oneDirection = false;

        perform_and_send_monitor_command(device, mutex, command);
        return static_cast<int8_t>(command.receivedCommandData[0]);
    }

    void get_f200_calibration_raw_data(uvc::device & device, std::timed_mutex & usbMutex, uint8_t * data, size_t & bytesReturned)
    {
        uint8_t request[IVCAM_MONITOR_HEADER_SIZE];
        size_t requestSize = sizeof(request);
        uint32_t responseOp;

        if (prepare_usb_command(request, requestSize, (uint32_t) IVCAMMonitorCommand::GetCalibrationTable) <= 0) 
            throw std::runtime_error("usb transfer to retrieve calibration data failed");
        execute_usb_command(device, usbMutex, request, requestSize, responseOp, data, bytesReturned);
    }

    void get_sr300_calibration_raw_data(uvc::device & device, std::timed_mutex & mutex, uint8_t * data, size_t & bytesReturned)
    {
        IVCAMCommand command(IVCAMMonitorCommand::GetCalibrationTable);
        command.Param1 = (uint32_t)IVCAMDataSource::TakeFromRAM;

        perform_and_send_monitor_command(device, mutex, command);
        memcpy(data, command.receivedCommandData, HW_MONITOR_BUFFER_SIZE);
        bytesReturned = command.receivedCommandDataLength;
    }
    
    int bcdtoint(uint8_t * buf, int bufsize)
    {
        int r = 0;
        for(int i = 0; i < bufsize; i++)
            r = r * 10 + *buf++;
        return r;
    }

    int get_version_of_calibration(uint8_t * validation, uint8_t * version)
    {
        uint8_t valid[2] = {0X14, 0x0A};
        if (memcmp(valid, validation, 2) != 0) return 0;
        else return bcdtoint(version, 2);
    }

    std::tuple<CameraCalibrationParameters, IVCAMTemperatureData, IVCAMThermalLoopParams> get_f200_calibration(uint8_t * rawCalibData, int len)
    {
        uint8_t * bufParams = rawCalibData + 4;

        IVCAMCalibration CalibrationData;
        IVCAMTesterData TesterData;

        memset(&CalibrationData, 0, sizeof(IVCAMCalibration));

        int ver = get_version_of_calibration(bufParams, bufParams + 2);

        if (ver > IVCAM_MIN_SUPPORTED_VERSION)
        {
            rawCalibData = rawCalibData + 4;

            int size = (sizeof(IVCAMCalibration) > len) ? len : sizeof(IVCAMCalibration);

            auto fixWithVersionInfo = [&](IVCAMCalibration &d, int size, uint8_t * data)
            {
                memcpy((uint8_t*)&d + sizeof(int), data, size - sizeof(int));
            };

            fixWithVersionInfo(CalibrationData, size, rawCalibData);

            CameraCalibrationParameters calibration;
            memcpy(&calibration, &CalibrationData.CalibrationParameters, sizeof(CameraCalibrationParameters));
            memcpy(&TesterData, rawCalibData, SIZE_OF_CALIB_HEADER_BYTES); //copy the header: valid + version

            //copy the tester data from end of calibration
            int EndOfCalibratioData = SIZE_OF_CALIB_PARAM_BYTES + SIZE_OF_CALIB_HEADER_BYTES;
            memcpy((uint8_t*)&TesterData + SIZE_OF_CALIB_HEADER_BYTES, rawCalibData + EndOfCalibratioData, sizeof(IVCAMTesterData) - SIZE_OF_CALIB_HEADER_BYTES);
            return std::make_tuple(calibration, TesterData.TemperatureData, TesterData.ThermalLoopParams);
        }

        if (ver == IVCAM_MIN_SUPPORTED_VERSION)
        {
            float *params = (float *)bufParams;

            CameraCalibrationParameters calibration;
            memcpy(&calibration, params + 1, sizeof(CameraCalibrationParameters)); // skip the first float or 2 uint16
            memcpy(&TesterData, bufParams, SIZE_OF_CALIB_HEADER_BYTES);

            memset((uint8_t*) &TesterData + SIZE_OF_CALIB_HEADER_BYTES, 0, sizeof(IVCAMTesterData) - SIZE_OF_CALIB_HEADER_BYTES);
            return std::make_tuple(calibration, TesterData.TemperatureData, TesterData.ThermalLoopParams);
        }

        throw std::runtime_error("calibration table is not compatible with this API");
    }
    
    std::tuple<CameraCalibrationParameters, IVCAMTemperatureData, IVCAMThermalLoopParams> read_f200_calibration(uvc::device & device, std::timed_mutex & mutex)
    {
        uint8_t rawCalibrationBuffer[HW_MONITOR_BUFFER_SIZE];
        size_t bufferLength = HW_MONITOR_BUFFER_SIZE;       
        get_f200_calibration_raw_data(device, mutex, rawCalibrationBuffer, bufferLength);
        return get_f200_calibration(rawCalibrationBuffer, bufferLength);
    }

    std::tuple<CameraCalibrationParameters, IVCAMTemperatureData, IVCAMThermalLoopParams> read_sr300_calibration(uvc::device & device, std::timed_mutex & mutex)
    {
        uint8_t rawCalibrationBuffer[HW_MONITOR_BUFFER_SIZE];
        size_t bufferLength = HW_MONITOR_BUFFER_SIZE;       
        get_sr300_calibration_raw_data(device, mutex, rawCalibrationBuffer, bufferLength);

        SR300RawCalibration rawCalib;
        memcpy(&rawCalib, rawCalibrationBuffer, std::min(sizeof(rawCalib), bufferLength)); // Is this longer or shorter than the rawCalib struct?
        IVCAMThermalLoopParams tlp;
        tlp.IRThermalLoopEnable = 0; // No need for SW thermal loop on SR300
        return std::make_tuple(rawCalib.CalibrationParameters, rawCalib.TemperatureData, tlp);
    }

    void generate_asic_calibration_coefficients(const CameraCalibrationParameters & compensated_calibration, std::vector<int> resolution, const bool isZMode, float * values)
    {
        auto params = compensated_calibration;

        //handle vertical crop at 360p - change to full 640x480 and crop at the end
        bool isQres = resolution[0] == 640 && resolution[1] == 360;

        if (isQres)
        {
            resolution[1] = 480;
        }

        //generate coefficients
        int scale = 5;

        float width = (float) resolution[0] * scale;
        float height = (float) resolution[1];

        int PrecisionBits = 16;
        int CodeBits = 14;
        int TexturePrecisionBits = 12;
        float ypscale = (float)(1<<(CodeBits+1-10));
        float ypoff = 0;

        float s1 = (float)(1<<(PrecisionBits)) / 2047; // (range max)
        float s2 = (float)(1<<(CodeBits))-ypscale*0.5f;

        float alpha = 2/(width*params.Kc[0][0]);
        float beta  = -(params.Kc[0][2]+1)/params.Kc[0][0];
        float gamma = 2/(height*params.Kc[1][1]);
        float delta = -(params.Kc[1][2]+1)/params.Kc[1][1];

        float a  = alpha/gamma;
        float a1 = 1;
        float b  = 0.5f*scale*a  + beta/gamma;
        float c  = 0.5f*a1 + delta/gamma;

        float d0 = 1;                 
        float d1 = params.Invdistc[0]*pow(gamma,(float)2.0);  
        float d2 = params.Invdistc[1]*pow(gamma,(float)4.0);  
        float d5 = (float)( (double)(params.Invdistc[4]) * pow((double)gamma,6.0) );  
        float d3 = params.Invdistc[2]*gamma;
        float d4 = params.Invdistc[3]*gamma;

        float q  = 1/pow(gamma,(float)2.0);
        float p1 = params.Pp[2][3]*s1;
        float p2 = -s1*s2*(params.Pp[1][3]+params.Pp[2][3]);

        if (isZMode)
        {
            p1 = p1*sqrt(q);
            p2 = p2*sqrt(q);
        }

        float p3 = -params.Pp[2][0];
        float p4 = -params.Pp[2][1];
        float p5 = -params.Pp[2][2]/gamma;
        float p6 = s2*(params.Pp[1][0]+params.Pp[2][0]);
        float p7 = s2*(params.Pp[1][1]+params.Pp[2][1]);
        float p8 = s2*(params.Pp[1][2]+params.Pp[2][2])/gamma;

        //Reprojection parameters
        float sreproj = 2;
        float ax = -(1+params.Kp[0][2])/params.Kp[0][0];
        float ay = -(1+params.Kp[1][2])/params.Kp[1][1];

        float f0 = (params.Pp[0][1]+params.Pp[2][1]) / (params.Pp[0][0]+params.Pp[2][0]) / params.Kp[0][0]; 
        float f1 = (params.Pp[0][2]+params.Pp[2][2]) / (params.Pp[0][0]+params.Pp[2][0]) / gamma / params.Kp[0][0];
        float f2 = 0; //(Pp(2,1)+Pp(3,1)) / (Pp(1,1)+Pp(3,1)) / Kp(5);
        float f3 = 0; //(Pp(2,2)+Pp(3,2)) / (Pp(1,1)+Pp(3,1)) / Kp(5);
        float f4 = 0; //(Pp(2,3)+Pp(3,3)) / (Pp(1,1)+Pp(3,1)) / gamma / Kp(5);
        float f5 = 2*params.Pp[2][0] / (params.Pp[0][0]+params.Pp[2][0]) / sreproj;
        float f6 = 2*params.Pp[2][1] / (params.Pp[0][0]+params.Pp[2][0]) / sreproj;
        float f7 = 2*params.Pp[2][2] / (params.Pp[0][0]+params.Pp[2][0]) / sreproj / gamma;
        float f8 = (float) ( (double)(params.Pp[0][3] + params.Pp[2][3]) / (params.Pp[0][0]+params.Pp[2][0]) * s1 / params.Kp[0][0] );
        float f9 = (params.Pp[1][3] + params.Pp[2][3]) / (params.Pp[0][0]+params.Pp[2][0]) * s1 / params.Kp[1][1];
        float f10 = 2*params.Pp[2][3] / (params.Pp[0][0]+params.Pp[2][0]) * s1 / sreproj;
        if (isZMode)
        {
            f8  = f8  * sqrt(q);
            f9  = 0; //f9  * sqrt(q);
            f10 = f10 * sqrt(q);
        }
        float f11 = 1 / params.Kp[0][0];

        // Fix projection center 
        f11 = f11 + ax*f5;
        f0  = f0  + ax*f6;
        f1  = f1  + ax*f7;
        f8  = f8  + ax*f10;
        f2  = f2  + ay*f5;
        f3  = f3  + ay*f6;
        f4  = f4  + ay*f7;
        f9  = f9  + ay*f10;

        // Texture coeffs
        float suv = (float) ( (1<<TexturePrecisionBits)-1 );

        float h0 =  (params.Pt[0][1] +  params.Pt[2][1]) / (params.Pt[0][0]+params.Pt[2][0]); 
        float h1 =  (params.Pt[0][2] +  params.Pt[2][2]) / (params.Pt[0][0]+params.Pt[2][0]) / gamma;
        float h2 =  (params.Pt[1][0] +  params.Pt[2][0]) / (params.Pt[0][0]+params.Pt[2][0]);
        float h3 =  (params.Pt[1][1] +  params.Pt[2][1]) / (params.Pt[0][0]+params.Pt[2][0]);
        float h4 =  (params.Pt[1][2] +  params.Pt[2][2]) / (params.Pt[0][0]+params.Pt[2][0]) / gamma;
        float h5 = 2*params.Pt[2][0] / (params.Pt[0][0]  +  params.Pt[2][0]) / suv;
        float h6 = 2*params.Pt[2][1] / (params.Pt[0][0]  +  params.Pt[2][0]) / suv;
        float h7 = 2*params.Pt[2][2] / (params.Pt[0][0]  +  params.Pt[2][0]) / suv / gamma;
        float h8 =  (params.Pt[0][3] +  params.Pt[2][3]) / (params.Pt[0][0]+params.Pt[2][0]) * s1;
        float h9 =  (params.Pt[1][3] +  params.Pt[2][3]) / (params.Pt[0][0]+params.Pt[2][0]) * s1;
        float h10 = 2*params.Pt[2][3] / (params.Pt[0][0]+params.Pt[2][0]) * s1 / suv;
        float h11 = 1;

        if (isZMode)
        {
            h8  = h8  * sqrt(q);
            h9  = h9  * sqrt(q);
            h10 = h10 * sqrt(q);
        }

        float o1 =  (1+params.Kp[0][2])/params.Kp[0][0];
        float o2 = -(1+params.Kp[1][2])/params.Kp[1][1];
        float o3 = 1/s2/params.Kp[1][1];
        float o4 = 0; //s2*(1+Kp(8));

        float dp1 = params.Distp[0];
        float dp2 = params.Distp[1];
        float dp3 = params.Distp[2];
        float dp4 = params.Distp[3];
        float dp5 = params.Distp[4];

        float ip0 = params.Kp[1][1]*s2;
        float ip1 = (s2*params.Invdistp[0]*params.Kp[1][1]) + s2*(1+params.Kp[1][2]);
        float ip2 = (s2*params.Invdistp[1]*params.Kp[1][1]);
        float ip3 = (s2*params.Invdistp[2]*params.Kp[1][1]);
        float ip4 = (s2*params.Invdistp[3]*params.Kp[1][1]);
        float ip5 = (s2*params.Invdistp[4]*params.Kp[1][1]);

        if (isQres)
            c *= 0.75;

        //to simplify the ordering of the coefficients, initialize it in list syntax and copy to allocated memory.
        float coeffs [NUM_OF_CALIBRATION_COEFFS] = {1.0f,3.0f,a,a1,b,c,d0,d1,d2,
            d3,d4,d5,q,p1,p2,p3,p4,p5,p6,p7,p8,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,f0,f1,
            f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,o1,o2,o3,o4,dp1,dp2,dp3,dp4,dp5,ip0,ip1,ip2,ip3,
            ip4,ip5,ypscale,ypoff,0,0};

        memcpy(values, coeffs, NUM_OF_CALIBRATION_COEFFS  * sizeof(float));
        return ;
    }
    
    void update_asic_coefficients(uvc::device & device, std::timed_mutex & mutex, const CameraCalibrationParameters & compensated_params)
    {
        IVCAMASICCoefficients coeffs = {};
        generate_asic_calibration_coefficients(compensated_params, {640, 480}, true, coeffs.CoefValueArray); // todo - fix hardcoded resolution parameters
        set_asic_coefficients(device, mutex, coeffs);    
    }    

}
    namespace sr300
    {
        void set_wakeup_device(uvc::device & device, std::timed_mutex & mutex, const uint32_t& phase1Period, const uint32_t& phase1FPS, const uint32_t& phase2Period, const uint32_t& phase2FPS)
        {
            f200::IVCAMCommand cmd(f200::IVCAMMonitorCommand::OnSuspendResume);

            sr300::wakeup_dev_params params = { phase1Period, static_cast<sr300::e_suspend_fps>(phase1FPS), phase2Period, static_cast<sr300::e_suspend_fps>(phase2FPS) };
            if (!params.isValid())
                throw std::logic_error("missing/invalid wake_up command parameters");
            cmd.Param1 = 1;                                                                 // TODO Specification could not be found in IVCAM
            auto trg = reinterpret_cast<sr300::wakeup_dev_params*>(cmd.data);
            *trg = params;
            cmd.sizeOfSendCommandData = sizeof(sr300::wakeup_dev_params);

            perform_and_send_monitor_command(device, mutex, cmd);
        }

        void reset_wakeup_device(uvc::device & device, std::timed_mutex & mutex)
        {
            f200::IVCAMCommand cmd(f200::IVCAMMonitorCommand::OnSuspendResume);

            perform_and_send_monitor_command(device, mutex, cmd);
        }

        void get_wakeup_reason(uvc::device & device, std::timed_mutex & mutex, unsigned char &cReason)
        {
            f200::IVCAMCommand cmdWUReason(f200::IVCAMMonitorCommand::GetWakeReason);

            perform_and_send_monitor_command(device, mutex, cmdWUReason);

            if (cmdWUReason.receivedCommandDataLength >= 4)     // TODO - better guard condition ?
            {
                unsigned char rslt = (*reinterpret_cast<int32_t *>(cmdWUReason.receivedCommandData)) && (0xFF);
                if (rslt >= (uint8_t)wakeonusb_reason::eMaxWakeOnReason)
                    throw std::logic_error("undefined wakeonusb_reason provided");
                cReason = rslt;
            }
            else
                throw std::runtime_error("no valid wakeonusb_reason provided");
        }

        void get_wakeup_confidence(uvc::device & device, std::timed_mutex & mutex, unsigned char &cConfidence)
        {
            f200::IVCAMCommand cmdCnfd(f200::IVCAMMonitorCommand::GetWakeConfidence);
            perform_and_send_monitor_command(device, mutex, cmdCnfd);

            if (cmdCnfd.receivedCommandDataLength >= 4)
            {
                int32_t rslt = *reinterpret_cast<int32_t *>(cmdCnfd.receivedCommandData);
                cConfidence = (unsigned char)(rslt & 0xFF);
            }
            else
                throw std::runtime_error("no valid wakeonusb_confidence provided");
        }

    } // namespace rsimpl::sr300
} // namespace rsimpl
