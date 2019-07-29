// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <k4a/k4a.h>

#include "assert.h"

#include "Open3D/IO/Sensor/cmdparser.h"
#include "Open3D/IO/Sensor/recorder.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <math.h>
#include <atomic>
#include <csignal>
#include <ctime>
#include <iostream>

static time_t exiting_timestamp;

static void signal_handler(int s) {
    (void)s;  // Unused

    if (!exiting) {
        std::cout << "Stopping recording..." << std::endl;
        exiting_timestamp = clock();
        exiting = true;
    }
    // If Ctrl-C is received again after 1 second, force-stop the application
    // since it's not responding.
    else if (exiting_timestamp != 0 &&
             clock() - exiting_timestamp > CLOCKS_PER_SEC) {
        std::cout << "Forcing stop." << std::endl;
        exit(1);
    }
}

static int string_compare(const char *s1, const char *s2) {
    assert(s1 != NULL);
    assert(s2 != NULL);

    while (tolower((unsigned char)*s1) == tolower((unsigned char)*s2)) {
        if (*s1 == '\0') {
            return 0;
        }
        s1++;
        s2++;
    }
    // The return value shows the relations between s1 and s2.
    // Return value   Description
    //     < 0        s1 less than s2
    //       0        s1 identical to s2
    //     > 0        s1 greater than s2
    return (int)tolower((unsigned char)*s1) - (int)tolower((unsigned char)*s2);
}

[[noreturn]] static void list_devices() {
    uint32_t device_count = k4a_device_get_installed_count();
    if (device_count > 0) {
        for (uint8_t i = 0; i < device_count; i++) {
            std::cout << "Index:" << (int)i;
            k4a_device_t device;
            if (K4A_SUCCEEDED(k4a_device_open(i, &device))) {
                char serial_number_buffer[256];
                size_t serial_number_buffer_size = sizeof(serial_number_buffer);
                if (k4a_device_get_serialnum(device, serial_number_buffer,
                                             &serial_number_buffer_size) ==
                    K4A_BUFFER_RESULT_SUCCEEDED) {
                    std::cout << "\tSerial:" << serial_number_buffer;
                } else {
                    std::cout << "\tSerial:ERROR";
                }

                k4a_hardware_version_t version_info;
                if (K4A_SUCCEEDED(
                            k4a_device_get_version(device, &version_info))) {
                    std::cout << "\tColor:" << version_info.rgb.major << "."
                              << version_info.rgb.minor << "."
                              << version_info.rgb.iteration;
                    std::cout << "\tDepth:" << version_info.depth.major << "."
                              << version_info.depth.minor << "."
                              << version_info.depth.iteration;
                }
                k4a_device_close(device);
            } else {
                std::cout << i << "\tDevice Open Failed";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "No devices connected." << std::endl;
    }
    exit(0);
}

int main(int argc, char **argv) {
    int device_index = 0;
    int recording_length = -1;
    k4a_image_format_t recording_color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    k4a_color_resolution_t recording_color_resolution =
            K4A_COLOR_RESOLUTION_1080P;
    k4a_depth_mode_t recording_depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    k4a_fps_t recording_rate = K4A_FRAMES_PER_SECOND_30;
    bool recording_rate_set = false;
    bool recording_imu_enabled = true;
    k4a_wired_sync_mode_t wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    int32_t depth_delay_off_color_usec = 0;
    uint32_t subordinate_delay_off_master_usec = 0;
    int absoluteExposureValue = 0;
    char *recording_filename;

    CmdParser::OptionParser cmd_parser;
    cmd_parser.RegisterOption("-h|--help", "Prints this help", [&]() {
        std::cout << "k4arecorder [options] output.mkv" << std::endl
                  << std::endl;
        cmd_parser.PrintOptions();
        exit(0);
    });
    cmd_parser.RegisterOption(
            "--list", "List the currently connected K4A devices", list_devices);
    cmd_parser.RegisterOption(
            "--device", "Specify the device index to use (default: 0)", 1,
            [&](const std::vector<char *> &args) {
                device_index = std::stoi(args[0]);
                if (device_index < 0 || device_index > 255)
                    throw std::runtime_error("Device index must 0-255");
            });
    cmd_parser.RegisterOption(
            "-l|--record-length",
            "Limit the recording to N seconds (default: infinite)", 1,
            [&](const std::vector<char *> &args) {
                recording_length = std::stoi(args[0]);
                if (recording_length < 0)
                    throw std::runtime_error(
                            "Recording length must be positive");
            });
    cmd_parser.RegisterOption(
            "-c|--color-mode",
            "Set the color sensor mode (default: 1080p), Available options:\n"
            "3072p, 2160p, 1536p, 1440p, 1080p, 720p, 720p_NV12, 720p_YUY2, "
            "OFF",
            1, [&](const std::vector<char *> &args) {
                if (string_compare(args[0], "3072p") == 0) {
                    recording_color_resolution = K4A_COLOR_RESOLUTION_3072P;
                } else if (string_compare(args[0], "2160p") == 0) {
                    recording_color_resolution = K4A_COLOR_RESOLUTION_2160P;
                } else if (string_compare(args[0], "1536p") == 0) {
                    recording_color_resolution = K4A_COLOR_RESOLUTION_1536P;
                } else if (string_compare(args[0], "1440p") == 0) {
                    recording_color_resolution = K4A_COLOR_RESOLUTION_1440P;
                } else if (string_compare(args[0], "1080p") == 0) {
                    recording_color_resolution = K4A_COLOR_RESOLUTION_1080P;
                } else if (string_compare(args[0], "720p") == 0) {
                    recording_color_resolution = K4A_COLOR_RESOLUTION_720P;
                } else if (string_compare(args[0], "720p_NV12") == 0) {
                    recording_color_format = K4A_IMAGE_FORMAT_COLOR_NV12;
                    recording_color_resolution = K4A_COLOR_RESOLUTION_720P;
                } else if (string_compare(args[0], "720p_YUY2") == 0) {
                    recording_color_format = K4A_IMAGE_FORMAT_COLOR_YUY2;
                    recording_color_resolution = K4A_COLOR_RESOLUTION_720P;
                } else if (string_compare(args[0], "off") == 0) {
                    recording_color_resolution = K4A_COLOR_RESOLUTION_OFF;
                } else {
                    recording_color_resolution = K4A_COLOR_RESOLUTION_OFF;

                    std::ostringstream str;
                    str << "Unknown color mode specified: " << args[0];
                    throw std::runtime_error(str.str());
                }
            });
    cmd_parser.RegisterOption(
            "-d|--depth-mode",
            "Set the depth sensor mode (default: NFOV_UNBINNED), Available "
            "options:\n"
            "NFOV_2X2BINNED, NFOV_UNBINNED, WFOV_2X2BINNED, WFOV_UNBINNED, "
            "PASSIVE_IR, OFF",
            1, [&](const std::vector<char *> &args) {
                if (string_compare(args[0], "NFOV_2X2BINNED") == 0) {
                    recording_depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
                } else if (string_compare(args[0], "NFOV_UNBINNED") == 0) {
                    recording_depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
                } else if (string_compare(args[0], "WFOV_2X2BINNED") == 0) {
                    recording_depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
                } else if (string_compare(args[0], "WFOV_UNBINNED") == 0) {
                    recording_depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
                } else if (string_compare(args[0], "PASSIVE_IR") == 0) {
                    recording_depth_mode = K4A_DEPTH_MODE_PASSIVE_IR;
                } else if (string_compare(args[0], "off") == 0) {
                    recording_depth_mode = K4A_DEPTH_MODE_OFF;
                } else {
                    std::ostringstream str;
                    str << "Unknown depth mode specified: " << args[0];
                    throw std::runtime_error(str.str());
                }
            });
    cmd_parser.RegisterOption("--depth-delay",
                              "Set the time offset between color and depth "
                              "frames in microseconds (default: 0)\n"
                              "A negative value means depth frames will arrive "
                              "before color frames.\n"
                              "The delay must be less than 1 frame period.",
                              1, [&](const std::vector<char *> &args) {
                                  depth_delay_off_color_usec =
                                          std::stoi(args[0]);
                              });
    cmd_parser.RegisterOption(
            "-r|--rate",
            "Set the camera frame rate in Frames per Second\n"
            "Default is the maximum rate supported by the camera modes.\n"
            "Available options: 30, 15, 5",
            1, [&](const std::vector<char *> &args) {
                recording_rate_set = true;
                if (string_compare(args[0], "30") == 0) {
                    recording_rate = K4A_FRAMES_PER_SECOND_30;
                } else if (string_compare(args[0], "15") == 0) {
                    recording_rate = K4A_FRAMES_PER_SECOND_15;
                } else if (string_compare(args[0], "5") == 0) {
                    recording_rate = K4A_FRAMES_PER_SECOND_5;
                } else {
                    std::ostringstream str;
                    str << "Unknown frame rate specified: " << args[0];
                    throw std::runtime_error(str.str());
                }
            });
    cmd_parser.RegisterOption(
            "--imu", "Set the IMU recording mode (ON, OFF, default: ON)", 1,
            [&](const std::vector<char *> &args) {
                if (string_compare(args[0], "on") == 0) {
                    recording_imu_enabled = true;
                } else if (string_compare(args[0], "off") == 0) {
                    recording_imu_enabled = false;
                } else {
                    std::ostringstream str;
                    str << "Unknown imu mode specified: " << args[0];
                    throw std::runtime_error(str.str());
                }
            });
    cmd_parser.RegisterOption(
            "--external-sync",
            "Set the external sync mode (Master, Subordinate, Standalone "
            "default: Standalone)",
            1, [&](const std::vector<char *> &args) {
                if (string_compare(args[0], "master") == 0) {
                    wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
                } else if (string_compare(args[0], "subordinate") == 0 ||
                           string_compare(args[0], "sub") == 0) {
                    wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
                } else if (string_compare(args[0], "standalone") == 0) {
                    wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
                } else {
                    std::ostringstream str;
                    str << "Unknown external sync mode specified: " << args[0];
                    throw std::runtime_error(str.str());
                }
            });
    cmd_parser.RegisterOption(
            "--sync-delay",
            "Set the external sync delay off the master camera in microseconds "
            "(default: 0)\n"
            "This setting is only valid if the camera is in Subordinate mode.",
            1, [&](const std::vector<char *> &args) {
                int delay = std::stoi(args[0]);
                if (delay < 0) {
                    throw std::runtime_error(
                            "External sync delay must be positive.");
                }
                subordinate_delay_off_master_usec = (uint32_t)delay;
            });
    cmd_parser.RegisterOption(
            "-e|--exposure-control",
            "Set manual exposure value (-11 to 1) for the RGB camera (default: "
            "auto exposure)",
            1, [&](const std::vector<char *> &args) {
                int exposureValue = std::stoi(args[0]);
                if (exposureValue < -11 || exposureValue > 1) {
                    throw std::runtime_error(
                            "Exposure value range is -11 to 1.");
                }
                absoluteExposureValue = static_cast<int32_t>(
                        exp2f((float)exposureValue) * 1000000.0f);
            });

    int args_left = 0;
    try {
        args_left = cmd_parser.ParseCmd(argc, argv);
    } catch (CmdParser::ArgumentError &e) {
        std::cerr << e.option() << ": " << e.what() << std::endl;
        return 1;
    }
    if (args_left == 1) {
        recording_filename = argv[argc - 1];
    } else {
        std::cout << "k4arecorder [options] output.mkv" << std::endl
                  << std::endl;
        cmd_parser.PrintOptions();
        return 0;
    }

    if (recording_rate == K4A_FRAMES_PER_SECOND_30 &&
        (recording_depth_mode == K4A_DEPTH_MODE_WFOV_UNBINNED ||
         recording_color_resolution == K4A_COLOR_RESOLUTION_3072P)) {
        if (!recording_rate_set) {
            // Default to max supported frame rate
            recording_rate = K4A_FRAMES_PER_SECOND_15;
        } else {
            std::cerr << "Error: 30 Frames per second is not supported by this "
                         "camera mode."
                      << std::endl;
            return 1;
        }
    }
    if (subordinate_delay_off_master_usec > 0 &&
        wired_sync_mode != K4A_WIRED_SYNC_MODE_SUBORDINATE) {
        std::cerr << "--sync-delay is only valid if --external-sync is set to "
                     "Subordinate."
                  << std::endl;
        return 1;
    }

#if defined(_WIN32)
    SetConsoleCtrlHandler(
            [](DWORD event) {
                if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) {
                    signal_handler(0);
                    return TRUE;
                }
                return FALSE;
            },
            true);
#else
    struct sigaction act;
    act.sa_handler = signal_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGINT, &act, 0);
#endif

    k4a_device_configuration_t device_config =
            K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    device_config.color_format = recording_color_format;
    device_config.color_resolution = recording_color_resolution;
    device_config.depth_mode = recording_depth_mode;
    device_config.camera_fps = recording_rate;
    device_config.wired_sync_mode = wired_sync_mode;
    device_config.depth_delay_off_color_usec = depth_delay_off_color_usec;
    device_config.subordinate_delay_off_master_usec =
            subordinate_delay_off_master_usec;

    return do_recording((uint8_t)device_index, recording_filename,
                        recording_length, &device_config, recording_imu_enabled,
                        absoluteExposureValue);
}
