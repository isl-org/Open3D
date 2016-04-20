// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "r200-private.h"

#include <cstring>
#include <cmath>
#include <ctime>
#include <thread>
#include <iomanip>

#pragma pack(push, 1) // All structs in this file are byte-aligend

enum class command : uint32_t // Command/response codes
{
    peek               = 0x11,
    poke               = 0x12,
    download_spi_flash = 0x1A,
    get_fwrevision     = 0x21,
};

enum class command_modifier : uint32_t { direct = 0x10 }; // Command/response modifiers

#define SPI_FLASH_PAGE_SIZE_IN_BYTES                0x100
#define SPI_FLASH_SECTOR_SIZE_IN_BYTES              0x1000
#define SPI_FLASH_SIZE_IN_SECTORS                   256
#define SPI_FLASH_TOTAL_SIZE_IN_BYTES               (SPI_FLASH_SIZE_IN_SECTORS * SPI_FLASH_SECTOR_SIZE_IN_BYTES)
#define SPI_FLASH_PAGES_PER_SECTOR                  (SPI_FLASH_SECTOR_SIZE_IN_BYTES / SPI_FLASH_PAGE_SIZE_IN_BYTES)
#define SPI_FLASH_SECTORS_RESERVED_FOR_FIRMWARE     160
#define NV_NON_FIRMWARE_START                       (SPI_FLASH_SECTORS_RESERVED_FOR_FIRMWARE * SPI_FLASH_SECTOR_SIZE_IN_BYTES)
#define NV_ADMIN_DATA_N_ENTRIES                     9
#define NV_CALIBRATION_DATA_ADDRESS_INDEX           0
#define NV_NON_FIRMWARE_ROOT_ADDRESS                NV_NON_FIRMWARE_START
#define CAM_INFO_BLOCK_LEN 2048

namespace rsimpl { namespace r200
{
    const uvc::extension_unit lr_xu = {0, 2, 1, {0x18682d34, 0xdd2c, 0x4073, {0xad, 0x23, 0x72, 0x14, 0x73, 0x9a, 0x07, 0x4c}}};

    void xu_read(const uvc::device & device, control xu_ctrl, void * buffer, uint32_t length)
    {
        uvc::get_control_with_retry(device, lr_xu, static_cast<int>(xu_ctrl), buffer, length);
    }

    void xu_write(uvc::device & device, control xu_ctrl, void * buffer, uint32_t length)
    {
        uvc::set_control_with_retry(device, lr_xu, static_cast<int>(xu_ctrl), buffer, length);
    }

    struct CommandResponsePacket
    {
        command code; command_modifier modifier;
        uint32_t tag, address, value, reserved[59];
        CommandResponsePacket() { std::memset(this, 0, sizeof(CommandResponsePacket)); }
        CommandResponsePacket(command code, uint32_t address=0, uint32_t value=0) : code(code), modifier(command_modifier::direct), tag(12), address(address), value(value)
        {
            std::memset(reserved, 0, sizeof(reserved));
        }
    };

    CommandResponsePacket send_command_and_receive_response(uvc::device & device, const CommandResponsePacket & command)
    {
        CommandResponsePacket c = command, r;
        set_control(device, lr_xu, static_cast<int>(control::command_response), &c, sizeof(c));
        get_control(device, lr_xu, static_cast<int>(control::command_response), &r, sizeof(r));
        return r;
    }

    bool read_device_pages(uvc::device & dev, uint32_t address, unsigned char * buffer, uint32_t nPages)
    {
        int addressTest = SPI_FLASH_TOTAL_SIZE_IN_BYTES - address - nPages * SPI_FLASH_PAGE_SIZE_IN_BYTES;

        if (!nPages || addressTest < 0)
            return false;

        // This command allows the host to read a block of data from the SPI flash.
        // Once this command is processed by the DS4, further command messages will be treated as SPI data
        // and therefore will be read from flash. The size of the SPI data must be a multiple of 256 bytes.
        // This will repeat until the number of bytes specified in the ‘value’ field of the original command
        // message has been read.  At that point the DS4 will process command messages as expected.

        send_command_and_receive_response(dev, CommandResponsePacket(command::download_spi_flash, address, nPages * SPI_FLASH_PAGE_SIZE_IN_BYTES));

        uint8_t *p = buffer;
        uint16_t spiLength = SPI_FLASH_PAGE_SIZE_IN_BYTES;
        for (unsigned int i = 0; i < nPages; ++i)
        {
            xu_read(dev, control::command_response, p, spiLength);
            p += SPI_FLASH_PAGE_SIZE_IN_BYTES;
        }
        return true;
    }

    void read_arbitrary_chunk(uvc::device & dev, uint32_t address, void * dataIn, int lengthInBytesIn)
    {
        unsigned char * data = (unsigned char *)dataIn;
        int lengthInBytes = lengthInBytesIn;
        unsigned char page[SPI_FLASH_PAGE_SIZE_IN_BYTES];
        int nPagesToRead;
        uint32_t startAddress = address;
        if (startAddress & 0xff)
        {
            // we are not on a page boundary
            startAddress = startAddress & ~0xff;
            uint32_t startInPage = address - startAddress;
            uint32_t lengthToCopy = SPI_FLASH_PAGE_SIZE_IN_BYTES - startInPage;
            if (lengthToCopy > (uint32_t)lengthInBytes)
                lengthToCopy = lengthInBytes;
            read_device_pages(dev, startAddress, page, 1);
            memcpy(data, page + startInPage, lengthToCopy);
            lengthInBytes -= lengthToCopy;
            data += lengthToCopy;
            startAddress += SPI_FLASH_PAGE_SIZE_IN_BYTES;
        }

        nPagesToRead = lengthInBytes / SPI_FLASH_PAGE_SIZE_IN_BYTES;

        if (nPagesToRead > 0)
            read_device_pages(dev, startAddress, data, nPagesToRead);

        lengthInBytes -= (nPagesToRead * SPI_FLASH_PAGE_SIZE_IN_BYTES);

        if (lengthInBytes)
        {
            // means we still have a remainder
            data += (nPagesToRead * SPI_FLASH_PAGE_SIZE_IN_BYTES);
            startAddress += (nPagesToRead * SPI_FLASH_PAGE_SIZE_IN_BYTES);
            read_device_pages(dev, startAddress, page, 1);
            memcpy(data, page, lengthInBytes);
        }
    }

    bool read_admin_sector(uvc::device & dev, unsigned char data[SPI_FLASH_SECTOR_SIZE_IN_BYTES], int whichAdminSector)
    {
        uint32_t adminSectorAddresses[NV_ADMIN_DATA_N_ENTRIES];

        read_arbitrary_chunk(dev, NV_NON_FIRMWARE_ROOT_ADDRESS, adminSectorAddresses, NV_ADMIN_DATA_N_ENTRIES * sizeof(adminSectorAddresses[0]));

        if (whichAdminSector >= 0 && whichAdminSector < NV_ADMIN_DATA_N_ENTRIES)
        {
            uint32_t pageAddressInBytes = adminSectorAddresses[whichAdminSector];
            return read_device_pages(dev, pageAddressInBytes, data, SPI_FLASH_PAGES_PER_SECTOR);
        }

        return false;
    }

    r200_calibration read_calibration_and_rectification_parameters(const uint8_t (& flash_data_buffer)[SPI_FLASH_SECTOR_SIZE_IN_BYTES])
    {
        struct RectifiedIntrinsics
        {
            big_endian<float> rfx, rfy;
            big_endian<float> rpx, rpy;
            big_endian<uint32_t> rw, rh;
            operator rs_intrinsics () const { return {(int)rw, (int)rh, rpx, rpy, rfx, rfy, RS_DISTORTION_NONE, {0,0,0,0,0}}; }
        };

        r200_calibration cameraCalib;
        cameraCalib.version = reinterpret_cast<const big_endian<uint32_t> &>(flash_data_buffer);
        if(cameraCalib.version == 0)
        {
            struct UnrectifiedIntrinsicsV0
            {
                big_endian<float> fx, fy;
                big_endian<float> px, py;
                big_endian<double> k[5];
                big_endian<uint32_t> w, h;
                operator rs_intrinsics () const { return {(int)w, (int)h, px, py, fx, fy, RS_DISTORTION_MODIFIED_BROWN_CONRADY, {(float)k[0],(float)k[1],(float)k[2],(float)k[3],(float)k[4]}}; }
            };

            struct CameraCalibrationParametersV0
            {
                enum { MAX_INTRIN_RIGHT = 2 };      ///< Max number right cameras supported (e.g. one or two, two would support a multi-baseline unit)
                enum { MAX_INTRIN_THIRD = 3 };      ///< Max number native resolutions the third camera can have (e.g. 1920x1080 and 640x480)
                enum { MAX_MODES_LR = 4 };    ///< Max number rectified LR resolution modes the structure supports (e.g. 640x480, 492x372 and 332x252)
                enum { MAX_MODES_THIRD = 4 }; ///< Max number rectified Third resolution modes the structure supports (e.g. 1920x1080, 1280x720, 640x480 and 320x240)

                big_endian<uint32_t> versionNumber;
                big_endian<uint16_t> numIntrinsicsRight;     ///< Number of right cameras < MAX_INTRIN_RIGHT_V0
                big_endian<uint16_t> numIntrinsicsThird;     ///< Number of native resolutions of third camera < MAX_INTRIN_THIRD_V0
                big_endian<uint16_t> numRectifiedModesLR;    ///< Number of rectified LR resolution modes < MAX_MODES_LR_V0
                big_endian<uint16_t> numRectifiedModesThird; ///< Number of rectified Third resolution modes < MAX_MODES_THIRD_V0

                UnrectifiedIntrinsicsV0 intrinsicsLeft;
                UnrectifiedIntrinsicsV0 intrinsicsRight[MAX_INTRIN_RIGHT];
                UnrectifiedIntrinsicsV0 intrinsicsThird[MAX_INTRIN_THIRD];

                RectifiedIntrinsics modesLR[MAX_INTRIN_RIGHT][MAX_MODES_LR];
                RectifiedIntrinsics modesThird[MAX_INTRIN_RIGHT][MAX_INTRIN_THIRD][MAX_MODES_THIRD];

                big_endian<double> Rleft[MAX_INTRIN_RIGHT][9];
                big_endian<double> Rright[MAX_INTRIN_RIGHT][9];
                big_endian<double> Rthird[MAX_INTRIN_RIGHT][9];

                big_endian<float> B[MAX_INTRIN_RIGHT];
                big_endian<float> T[MAX_INTRIN_RIGHT][3];

                big_endian<double> Rworld[9];
                big_endian<float> Tworld[3];
            };

            const auto & calib = reinterpret_cast<const CameraCalibrationParametersV0 &>(flash_data_buffer);
            for(int i=0; i<3; ++i) cameraCalib.modesLR[i] = calib.modesLR[0][i];
            for(int i=0; i<2; ++i)
            {
                cameraCalib.intrinsicsThird[i] = calib.intrinsicsThird[i];
                for(int j=0; j<2; ++j) cameraCalib.modesThird[i][j] = calib.modesThird[0][i][j];
            }
            for(int i=0; i<9; ++i) cameraCalib.Rthird[i] = static_cast<float>(calib.Rthird[0][i]);
            for(int i=0; i<3; ++i) cameraCalib.T[i] = calib.T[0][i];
            cameraCalib.B = calib.B[0];
        }
        else if(cameraCalib.version == 1 || cameraCalib.version == 2)
        {
            struct UnrectifiedIntrinsicsV2
            {
                big_endian<float> fx, fy;
                big_endian<float> px, py;
                big_endian<float> k[5];
                big_endian<uint32_t> w, h;
                operator rs_intrinsics () const { return {(int)w, (int)h, px, py, fx, fy, RS_DISTORTION_MODIFIED_BROWN_CONRADY, {k[0],k[1],k[2],k[3],k[4]}}; }
            };

            struct CameraCalibrationParametersV2
            {
                enum { MAX_INTRIN_RIGHT = 2 }; // Max number right cameras supported (e.g. one or two, two would support a multi-baseline unit)
                enum { MAX_INTRIN_THIRD = 3 }; // Max number native resolutions the third camera can have (e.g. 1920x1080 and 640x480)
                enum { MAX_INTRIN_PLATFORM = 4 }; // Max number native resolutions the platform camera can have
                enum { MAX_MODES_LR = 4 }; // Max number rectified LR resolution modes the structure supports (e.g. 640x480, 492x372 and 332x252)
                enum { MAX_MODES_THIRD = 3 }; // Max number rectified Third resolution modes the structure supports (e.g. 1920x1080, 1280x720, etc)
                enum { MAX_MODES_PLATFORM = 1 }; // Max number rectified Platform resolution modes the structure supports

                big_endian<uint32_t> versionNumber;
                big_endian<uint16_t> numIntrinsicsRight;
                big_endian<uint16_t> numIntrinsicsThird;
                big_endian<uint16_t> numIntrinsicsPlatform;
                big_endian<uint16_t> numRectifiedModesLR;
                big_endian<uint16_t> numRectifiedModesThird;
                big_endian<uint16_t> numRectifiedModesPlatform;

                UnrectifiedIntrinsicsV2 intrinsicsLeft;
                UnrectifiedIntrinsicsV2 intrinsicsRight[MAX_INTRIN_RIGHT];
                UnrectifiedIntrinsicsV2 intrinsicsThird[MAX_INTRIN_THIRD];
                UnrectifiedIntrinsicsV2 intrinsicsPlatform[MAX_INTRIN_PLATFORM];

                RectifiedIntrinsics modesLR[MAX_INTRIN_RIGHT][MAX_MODES_LR];
                RectifiedIntrinsics modesThird[MAX_INTRIN_RIGHT][MAX_INTRIN_THIRD][MAX_MODES_THIRD];
                RectifiedIntrinsics modesPlatform[MAX_INTRIN_RIGHT][MAX_INTRIN_PLATFORM][MAX_MODES_PLATFORM];

                big_endian<float> Rleft[MAX_INTRIN_RIGHT][9];
                big_endian<float> Rright[MAX_INTRIN_RIGHT][9];
                big_endian<float> Rthird[MAX_INTRIN_RIGHT][9];
                big_endian<float> Rplatform[MAX_INTRIN_RIGHT][9];

                big_endian<float> B[MAX_INTRIN_RIGHT];
                big_endian<float> T[MAX_INTRIN_RIGHT][3];
                big_endian<float> Tplatform[MAX_INTRIN_RIGHT][3];

                big_endian<float> Rworld[9];
                big_endian<float> Tworld[3];
            };

            const auto & calib = reinterpret_cast<const CameraCalibrationParametersV2 &>(flash_data_buffer);
            for(int i=0; i<3; ++i) cameraCalib.modesLR[i] = calib.modesLR[0][i];
            for(int i=0; i<2; ++i)
            {
                cameraCalib.intrinsicsThird[i] = calib.intrinsicsThird[i];
                for(int j=0; j<2; ++j) cameraCalib.modesThird[i][j] = calib.modesThird[0][i][j];
            }
            for(int i=0; i<9; ++i) cameraCalib.Rthird[i] = calib.Rthird[0][i];
            for(int i=0; i<3; ++i) cameraCalib.T[i] = calib.T[0][i];
            cameraCalib.B = calib.B[0];
        }
        else
        {
            throw std::runtime_error(to_string() << "Unsupported calibration version: " << cameraCalib.version);
        }

        return cameraCalib;
    }

    void read_camera_head_contents(const uint8_t (& flash_data_buffer)[SPI_FLASH_SECTOR_SIZE_IN_BYTES], uint32_t & serial_number)
    {
        struct CameraHeadContents
        {
            enum { VERSION_NUMBER = 12 };
            uint32_t serialNumber;
            uint32_t modelNumber;
            uint32_t revisionNumber;
            uint8_t modelData[64];
            double buildDate;
            double firstProgramDate;
            double focusAndAlignmentDate;
            uint32_t nominalBaselineThird;
            uint8_t moduleVersion;
            uint8_t moduleMajorVersion;
            uint8_t moduleMinorVersion;
            uint8_t moduleSkew;
            uint32_t lensTypeThird;
            uint32_t OEMID;
            uint32_t lensCoatingTypeThird;
            uint8_t platformCameraSupport;
            uint8_t reserved1[3];
            uint32_t emitterType;
            uint8_t reserved2[4];
            uint32_t cameraFPGAVersion;
            uint32_t platformCameraFocus; // This is the value during calibration
            double calibrationDate;
            uint32_t calibrationType;
            double calibrationXError;
            double calibrationYError;
            double rectificationDataQres[54];
            double rectificationDataPadding[26];
            double CxQres;
            double CyQres;
            double CzQres;
            double KxQres;
            double KyQres;
            uint32_t cameraHeadContentsVersion;
            uint32_t cameraHeadContentsSizeInBytes;
            double CxBig;
            double CyBig;
            double CzBig;
            double KxBig;
            double KyBig;
            double CxSpecial;
            double CySpecial;
            double CzSpecial;
            double KxSpecial;
            double KySpecial;
            uint8_t cameraHeadDataLittleEndian;
            double rectificationDataBig[54];
            double rectificationDataSpecial[54];
            uint8_t cameraOptions1;
            uint8_t cameraOptions2;
            uint8_t bodySerialNumber[20];
            double Dx;
            double Dy;
            double Dz;
            double ThetaX;
            double ThetaY;
            double ThetaZ;
            double registrationDate;
            double registrationRotation[9];
            double registrationTranslation[3];
            uint32_t nominalBaseline;
            uint32_t lensType;
            uint32_t lensCoating;
            int32_t nominalBaselinePlatform[3]; // NOTE: Signed, since platform camera can be mounted anywhere
            uint32_t lensTypePlatform;
            uint32_t imagerTypePlatform;
            uint32_t theLastWord;
            uint8_t reserved3[37];
        };

        auto header = reinterpret_cast<const CameraHeadContents &>(flash_data_buffer[CAM_INFO_BLOCK_LEN]);
        serial_number = header.serialNumber;

        auto build_date = time_t(header.buildDate), calib_date = time_t(header.calibrationDate);
        LOG_INFO("Serial number                       = " << header.serialNumber);
        LOG_INFO("Model number                        = " << header.modelNumber);
        LOG_INFO("Revision number                     = " << header.revisionNumber);
        LOG_INFO("Camera head contents version        = " << header.cameraHeadContentsVersion);
        if(header.cameraHeadContentsVersion != CameraHeadContents::VERSION_NUMBER) LOG_WARNING("Camera head contents version != 12, data may be missing/incorrect");
        LOG_INFO("Module version                      = " << (int)header.moduleVersion << "." << (int)header.moduleMajorVersion << "." << (int)header.moduleMinorVersion << "." << (int)header.moduleSkew);
        LOG_INFO("OEM ID                              = " << header.OEMID);        
        LOG_INFO("Lens type for left/right imagers    = " << header.lensType);
        LOG_INFO("Lens type for third imager          = " << header.lensTypeThird);
        LOG_INFO("Lens coating for left/right imagers = " << header.lensCoating);
        LOG_INFO("Lens coating for third imager       = " << header.lensCoatingTypeThird);
        LOG_INFO("Nominal baseline (left to right)    = " << header.nominalBaseline << " mm");
        LOG_INFO("Nominal baseline (left to third)    = " << header.nominalBaselineThird << " mm");
        //if(std::isfinite(header.buildDate)) LOG_INFO("Built on " << std::put_time(std::gmtime(&build_date), "%Y-%m-%d %H:%M:%S") << " UTC");
        //if(std::isfinite(header.calibrationDate)) LOG_INFO("Calibrated on " << std::put_time(std::gmtime(&calib_date), "%Y-%m-%d %H:%M:%S") << " UTC");
    }

    r200_calibration read_camera_info(uvc::device & device)
    {
        uint8_t flashDataBuffer[SPI_FLASH_SECTOR_SIZE_IN_BYTES];
        if(!read_admin_sector(device, flashDataBuffer, NV_CALIBRATION_DATA_ADDRESS_INDEX)) throw std::runtime_error("Could not read calibration sector");

        auto calib = read_calibration_and_rectification_parameters(flashDataBuffer);
        read_camera_head_contents(flashDataBuffer, calib.serial_number);
        return calib;
    }

    std::string read_firmware_version(uvc::device & device)
    {
        auto response = send_command_and_receive_response(device, CommandResponsePacket(command::get_fwrevision));
        return reinterpret_cast<const char *>(response.reserved);
    }

    void set_stream_intent(uvc::device & device, uint8_t & intent)
    {
        xu_write(device, control::stream_intent, intent);
    }

    void get_stream_status(const uvc::device & device, int & status)
    {
        uint8_t s[4] = {255, 255, 255, 255};
        xu_read(device, control::status, s, sizeof(uint32_t));
        status = rsimpl::pack(s[0], s[1], s[2], s[3]);
    }

    void force_firmware_reset(uvc::device & device)
    {
        try
        {
            uint8_t reset = 1;
            xu_write(device, control::sw_reset, &reset, sizeof(uint8_t));
        }
        catch(...) {} // xu_write always throws during a control::SW_RESET, since the firmware is unable to send a proper response
    }

    bool get_emitter_state(const uvc::device & device, bool is_streaming, bool is_depth_enabled)
    {
        auto byte = xu_read<uint8_t>(device, control::emitter);
        if(is_streaming) return (byte & 1 ? true : false);
        else if(byte & 4) return (byte & 2 ? true : false);
        else return is_depth_enabled;
    }

    void set_emitter_state(uvc::device & device, bool state)
    {
        xu_write(device, control::emitter, uint8_t(state ? 1 : 0));
    }

	void get_register_value(uvc::device & device, uint32_t reg, uint32_t & value)
    {
        value = send_command_and_receive_response(device, CommandResponsePacket(command::peek, reg)).value;
    }

	void set_register_value(uvc::device & device, uint32_t reg, uint32_t value)
    {
		send_command_and_receive_response(device, CommandResponsePacket(command::poke, reg, value));
    }

    const dc_params dc_params::presets[] = {
        {5, 5, 192,  1,  512, 6, 24, 27,  7,   24}, // (DEFAULT) Default settings on chip. Similiar to the medium setting and best for outdoors.
        {5, 5,   0,  0, 1023, 0,  0,  0,  0, 2047}, // (OFF) Disable almost all hardware-based outlier removal
        {5, 5, 115,  1,  512, 6, 18, 25,  3,   24}, // (LOW) Provide a depthmap with a lower number of outliers removed, which has minimal false negatives.
        {5, 5, 185,  5,  505, 6, 35, 45, 45,   14}, // (MEDIUM) Provide a depthmap with a medium number of outliers removed, which has balanced approach.
        {5, 5, 175, 24,  430, 6, 48, 47, 24,   12}, // (OPTIMIZED) Provide a depthmap with a medium/high number of outliers removed. Derived from an optimization function.
        {5, 5, 235, 27,  420, 8, 80, 70, 90,   12}, // (HIGH) Provide a depthmap with a higher number of outliers removed, which has minimal false positives.
    };

} } // namespace rsimpl::r200

#pragma pack(pop)
