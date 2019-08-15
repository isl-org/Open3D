
#ifndef K4ATYPES_H
#define K4ATYPES_H
#ifdef __cplusplus
#include <cinttypes>
#include <cstddef>
#include <cstring>
#else
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define K4A_DECLARE_HANDLE(_handle_name_) \
    typedef struct _##_handle_name_ {     \
        size_t _rsvd;                     \
    } * _handle_name_;
K4A_DECLARE_HANDLE(k4a_device_t);
K4A_DECLARE_HANDLE(k4a_capture_t);
K4A_DECLARE_HANDLE(k4a_image_t);
K4A_DECLARE_HANDLE(k4a_transformation_t);

typedef enum {
    K4A_RESULT_SUCCEEDED = 0,
    K4A_RESULT_FAILED,
} k4a_result_t;

typedef enum {
    K4A_BUFFER_RESULT_SUCCEEDED = 0,
    K4A_BUFFER_RESULT_FAILED,
    K4A_BUFFER_RESULT_TOO_SMALL,
} k4a_buffer_result_t;

typedef enum {
    K4A_WAIT_RESULT_SUCCEEDED = 0,
    K4A_WAIT_RESULT_FAILED,
    K4A_WAIT_RESULT_TIMEOUT,
} k4a_wait_result_t;

typedef enum {
    K4A_LOG_LEVEL_CRITICAL = 0,
    K4A_LOG_LEVEL_ERROR,
    K4A_LOG_LEVEL_WARNING,
    K4A_LOG_LEVEL_INFO,
    K4A_LOG_LEVEL_TRACE,
    K4A_LOG_LEVEL_OFF,
} k4a_log_level_t;

typedef enum {
    K4A_DEPTH_MODE_OFF,
    K4A_DEPTH_MODE_NFOV_2X2BINNED,
    K4A_DEPTH_MODE_NFOV_UNBINNED,
    K4A_DEPTH_MODE_WFOV_2X2BINNED,
    K4A_DEPTH_MODE_WFOV_UNBINNED,
    K4A_DEPTH_MODE_PASSIVE_IR,
} k4a_depth_mode_t;

typedef enum {
    K4A_COLOR_RESOLUTION_OFF,
    K4A_COLOR_RESOLUTION_720P,
    K4A_COLOR_RESOLUTION_1080P,
    K4A_COLOR_RESOLUTION_1440P,
    K4A_COLOR_RESOLUTION_1536P,
    K4A_COLOR_RESOLUTION_2160P,
    K4A_COLOR_RESOLUTION_3072P,
} k4a_color_resolution_t;

typedef enum {
    K4A_IMAGE_FORMAT_COLOR_MJPG,
    K4A_IMAGE_FORMAT_COLOR_NV12,
    K4A_IMAGE_FORMAT_COLOR_YUY2,
    K4A_IMAGE_FORMAT_COLOR_BGRA32,
    K4A_IMAGE_FORMAT_DEPTH16,
    K4A_IMAGE_FORMAT_IR16,
    K4A_IMAGE_FORMAT_CUSTOM,
} k4a_image_format_t;

typedef enum {
    K4A_FRAMES_PER_SECOND_5,
    K4A_FRAMES_PER_SECOND_15,
    K4A_FRAMES_PER_SECOND_30,
} k4a_fps_t;

typedef enum {
    K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE = 0,
    K4A_COLOR_CONTROL_AUTO_EXPOSURE_PRIORITY,
    K4A_COLOR_CONTROL_BRIGHTNESS,
    K4A_COLOR_CONTROL_CONTRAST,
    K4A_COLOR_CONTROL_SATURATION,
    K4A_COLOR_CONTROL_SHARPNESS,
    K4A_COLOR_CONTROL_WHITEBALANCE,
    K4A_COLOR_CONTROL_BACKLIGHT_COMPENSATION,
    K4A_COLOR_CONTROL_GAIN,
    K4A_COLOR_CONTROL_POWERLINE_FREQUENCY
} k4a_color_control_command_t;

typedef enum {
    K4A_COLOR_CONTROL_MODE_AUTO = 0,
    K4A_COLOR_CONTROL_MODE_MANUAL,
} k4a_color_control_mode_t;

typedef enum {
    K4A_WIRED_SYNC_MODE_STANDALONE,
    K4A_WIRED_SYNC_MODE_MASTER,
    K4A_WIRED_SYNC_MODE_SUBORDINATE,
} k4a_wired_sync_mode_t;

typedef enum {
    K4A_CALIBRATION_TYPE_UNKNOWN = -1,
    K4A_CALIBRATION_TYPE_DEPTH,
    K4A_CALIBRATION_TYPE_COLOR,
    K4A_CALIBRATION_TYPE_GYRO,
    K4A_CALIBRATION_TYPE_ACCEL,
    K4A_CALIBRATION_TYPE_NUM,
} k4a_calibration_type_t;

typedef enum {
    K4A_CALIBRATION_LENS_DISTORTION_MODEL_UNKNOWN = 0,
    K4A_CALIBRATION_LENS_DISTORTION_MODEL_THETA,
    K4A_CALIBRATION_LENS_DISTORTION_MODEL_POLYNOMIAL_3K,
    K4A_CALIBRATION_LENS_DISTORTION_MODEL_RATIONAL_6KT,
    K4A_CALIBRATION_LENS_DISTORTION_MODEL_BROWN_CONRADY,
} k4a_calibration_model_type_t;

typedef enum {
    K4A_FIRMWARE_BUILD_RELEASE,
    K4A_FIRMWARE_BUILD_DEBUG
} k4a_firmware_build_t;

typedef enum {
    K4A_FIRMWARE_SIGNATURE_MSFT,
    K4A_FIRMWARE_SIGNATURE_TEST,
    K4A_FIRMWARE_SIGNATURE_UNSIGNED
} k4a_firmware_signature_t;
#define K4A_SUCCEEDED(_result_) (_result_ == K4A_RESULT_SUCCEEDED)
#define K4A_FAILED(_result_) (!K4A_SUCCEEDED(_result_))

typedef void(k4a_logging_message_cb_t)(void *context,
                                       k4a_log_level_t level,
                                       const char *file,
                                       const int line,
                                       const char *message);

typedef void(k4a_memory_destroy_cb_t)(void *buffer, void *context);

typedef struct _k4a_device_configuration_t {
    k4a_image_format_t color_format;
    k4a_color_resolution_t color_resolution;
    k4a_depth_mode_t depth_mode;
    k4a_fps_t camera_fps;
    bool synchronized_images_only;
    int32_t depth_delay_off_color_usec;
    k4a_wired_sync_mode_t wired_sync_mode;
    uint32_t subordinate_delay_off_master_usec;
    bool disable_streaming_indicator;
} k4a_device_configuration_t;

typedef struct _k4a_calibration_extrinsics_t {
    float rotation[9];
    float translation[3];
} k4a_calibration_extrinsics_t;

typedef union {
    struct _param {
        float cx;
        float cy;
        float fx;
        float fy;
        float k1;
        float k2;
        float k3;
        float k4;
        float k5;
        float k6;
        float codx;
        float cody;
        float p2;
        float p1;
        float metric_radius;
    } param;
    float v[15];
} k4a_calibration_intrinsic_parameters_t;

typedef struct _k4a_calibration_intrinsics_t {
    k4a_calibration_model_type_t type;
    unsigned int parameter_count;
    k4a_calibration_intrinsic_parameters_t parameters;
} k4a_calibration_intrinsics_t;

typedef struct _k4a_calibration_camera_t {
    k4a_calibration_extrinsics_t extrinsics;
    k4a_calibration_intrinsics_t intrinsics;
    int resolution_width;
    int resolution_height;
    float metric_radius;
} k4a_calibration_camera_t;

typedef struct _k4a_calibration_t {
    k4a_calibration_camera_t depth_camera_calibration;
    k4a_calibration_camera_t color_camera_calibration;
    k4a_calibration_extrinsics_t extrinsics[K4A_CALIBRATION_TYPE_NUM]
                                           [K4A_CALIBRATION_TYPE_NUM];
    k4a_depth_mode_t depth_mode;
    k4a_color_resolution_t color_resolution;
} k4a_calibration_t;

typedef struct _k4a_version_t {
    uint32_t major;
    uint32_t minor;
    uint32_t iteration;
} k4a_version_t;

typedef struct _k4a_hardware_version_t {
    k4a_version_t rgb;
    k4a_version_t depth;
    k4a_version_t audio;
    k4a_version_t depth_sensor;
    k4a_firmware_build_t firmware_build;
    k4a_firmware_signature_t firmware_signature;
} k4a_hardware_version_t;

typedef union {
    struct _xy {
        float x;
        float y;
    } xy;
    float v[2];
} k4a_float2_t;

typedef union {
    struct _xyz {
        float x;
        float y;
        float z;
    } xyz;
    float v[3];
} k4a_float3_t;

typedef struct _k4a_imu_sample_t {
    float temperature;
    k4a_float3_t acc_sample;
    uint64_t acc_timestamp_usec;
    k4a_float3_t gyro_sample;
    uint64_t gyro_timestamp_usec;
} k4a_imu_sample_t;
#define K4A_DEVICE_DEFAULT (0)
#define K4A_WAIT_INFINITE (-1)
static const k4a_device_configuration_t K4A_DEVICE_CONFIG_INIT_DISABLE_ALL = {
        K4A_IMAGE_FORMAT_COLOR_MJPG,
        K4A_COLOR_RESOLUTION_OFF,
        K4A_DEPTH_MODE_OFF,
        K4A_FRAMES_PER_SECOND_30,
        false,
        0,
        K4A_WIRED_SYNC_MODE_STANDALONE,
        0,
        false};
#ifdef __cplusplus
}
#endif
#endif /* K4ATYPES_H */
