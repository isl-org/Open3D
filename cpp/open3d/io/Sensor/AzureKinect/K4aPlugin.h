// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

// To prevent header dependency propagation
// Do not include this file in any *.h, only in *.cc
#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>

namespace open3d {
namespace io {
namespace k4a_plugin {

k4a_result_t k4a_record_create(const char *path,
                               k4a_device_t device,
                               const k4a_device_configuration_t device_config,
                               k4a_record_t *recording_handle);

k4a_result_t k4a_record_add_tag(k4a_record_t recording_handle,
                                const char *name,
                                const char *value);

k4a_result_t k4a_record_add_imu_track(k4a_record_t recording_handle);

k4a_result_t k4a_record_write_header(k4a_record_t recording_handle);

k4a_result_t k4a_record_write_capture(k4a_record_t recording_handle,
                                      k4a_capture_t capture_handle);

k4a_result_t k4a_record_write_imu_sample(k4a_record_t recording_handle,
                                         k4a_imu_sample_t imu_sample);

k4a_result_t k4a_record_flush(k4a_record_t recording_handle);

void k4a_record_close(k4a_record_t recording_handle);

////////////////////////////////////////////////////////////////////////////////

k4a_result_t k4a_playback_open(const char *path,
                               k4a_playback_t *playback_handle);

k4a_buffer_result_t k4a_playback_get_raw_calibration(
        k4a_playback_t playback_handle, uint8_t *data, size_t *data_size);

k4a_result_t k4a_playback_get_calibration(k4a_playback_t playback_handle,
                                          k4a_calibration_t *calibration);

k4a_result_t k4a_playback_get_record_configuration(
        k4a_playback_t playback_handle, k4a_record_configuration_t *config);

k4a_buffer_result_t k4a_playback_get_tag(k4a_playback_t playback_handle,
                                         const char *name,
                                         char *value,
                                         size_t *value_size);

k4a_result_t k4a_playback_set_color_conversion(
        k4a_playback_t playback_handle, k4a_image_format_t target_format);

k4a_stream_result_t k4a_playback_get_next_capture(
        k4a_playback_t playback_handle, k4a_capture_t *capture_handle);

k4a_stream_result_t k4a_playback_get_previous_capture(
        k4a_playback_t playback_handle, k4a_capture_t *capture_handle);

k4a_stream_result_t k4a_playback_get_next_imu_sample(
        k4a_playback_t playback_handle, k4a_imu_sample_t *imu_sample);

k4a_stream_result_t k4a_playback_get_previous_imu_sample(
        k4a_playback_t playback_handle, k4a_imu_sample_t *imu_sample);

k4a_result_t k4a_playback_seek_timestamp(k4a_playback_t playback_handle,
                                         int64_t offset_usec,
                                         k4a_playback_seek_origin_t origin);

uint64_t k4a_playback_get_last_timestamp_usec(k4a_playback_t playback_handle);

void k4a_playback_close(k4a_playback_t playback_handle);

////////////////////////////////////////////////////////////////////////////////

uint32_t k4a_device_get_installed_count(void);

k4a_result_t k4a_set_debug_message_handler(k4a_logging_message_cb_t *message_cb,
                                           void *message_cb_context,
                                           k4a_log_level_t min_level);

k4a_result_t k4a_device_open(uint32_t index, k4a_device_t *device_handle);

void k4a_device_close(k4a_device_t device_handle);

k4a_wait_result_t k4a_device_get_capture(k4a_device_t device_handle,
                                         k4a_capture_t *capture_handle,
                                         int32_t timeout_in_ms);

k4a_wait_result_t k4a_device_get_imu_sample(k4a_device_t device_handle,
                                            k4a_imu_sample_t *imu_sample,
                                            int32_t timeout_in_ms);

k4a_result_t k4a_capture_create(k4a_capture_t *capture_handle);

void k4a_capture_release(k4a_capture_t capture_handle);

void k4a_capture_reference(k4a_capture_t capture_handle);

k4a_image_t k4a_capture_get_color_image(k4a_capture_t capture_handle);

k4a_image_t k4a_capture_get_depth_image(k4a_capture_t capture_handle);

k4a_image_t k4a_capture_get_ir_image(k4a_capture_t capture_handle);

void k4a_capture_set_color_image(k4a_capture_t capture_handle,
                                 k4a_image_t image_handle);

void k4a_capture_set_depth_image(k4a_capture_t capture_handle,
                                 k4a_image_t image_handle);

void k4a_capture_set_ir_image(k4a_capture_t capture_handle,
                              k4a_image_t image_handle);

void k4a_capture_set_temperature_c(k4a_capture_t capture_handle,
                                   float temperature_c);

float k4a_capture_get_temperature_c(k4a_capture_t capture_handle);

k4a_result_t k4a_image_create(k4a_image_format_t format,
                              int width_pixels,
                              int height_pixels,
                              int stride_bytes,
                              k4a_image_t *image_handle);

k4a_result_t k4a_image_create_from_buffer(
        k4a_image_format_t format,
        int width_pixels,
        int height_pixels,
        int stride_bytes,
        uint8_t *buffer,
        size_t buffer_size,
        k4a_memory_destroy_cb_t *buffer_release_cb,
        void *buffer_release_cb_context,
        k4a_image_t *image_handle);

uint8_t *k4a_image_get_buffer(k4a_image_t image_handle);

size_t k4a_image_get_size(k4a_image_t image_handle);

k4a_image_format_t k4a_image_get_format(k4a_image_t image_handle);

int k4a_image_get_width_pixels(k4a_image_t image_handle);

int k4a_image_get_height_pixels(k4a_image_t image_handle);

int k4a_image_get_stride_bytes(k4a_image_t image_handle);

uint64_t k4a_image_get_timestamp_usec(k4a_image_t image_handle);

uint64_t k4a_image_get_exposure_usec(k4a_image_t image_handle);

uint32_t k4a_image_get_white_balance(k4a_image_t image_handle);

uint32_t k4a_image_get_iso_speed(k4a_image_t image_handle);

void k4a_image_set_timestamp_usec(k4a_image_t image_handle,
                                  uint64_t timestamp_usec);

void k4a_image_set_exposure_time_usec(k4a_image_t image_handle,
                                      uint64_t exposure_usec);

void k4a_image_set_white_balance(k4a_image_t image_handle,
                                 uint32_t white_balance);

void k4a_image_set_iso_speed(k4a_image_t image_handle, uint32_t iso_speed);

void k4a_image_reference(k4a_image_t image_handle);

void k4a_image_release(k4a_image_t image_handle);

k4a_result_t k4a_device_start_cameras(k4a_device_t device_handle,
                                      k4a_device_configuration_t *config);

void k4a_device_stop_cameras(k4a_device_t device_handle);

k4a_result_t k4a_device_start_imu(k4a_device_t device_handle);

void k4a_device_stop_imu(k4a_device_t device_handle);

k4a_buffer_result_t k4a_device_get_serialnum(k4a_device_t device_handle,
                                             char *serial_number,
                                             size_t *serial_number_size);

k4a_result_t k4a_device_get_version(k4a_device_t device_handle,
                                    k4a_hardware_version_t *version);

k4a_result_t k4a_device_get_color_control_capabilities(
        k4a_device_t device_handle,
        k4a_color_control_command_t command,
        bool *supports_auto,
        int32_t *min_value,
        int32_t *max_value,
        int32_t *step_value,
        int32_t *default_value,
        k4a_color_control_mode_t *default_mode);

k4a_result_t k4a_device_get_color_control(k4a_device_t device_handle,
                                          k4a_color_control_command_t command,
                                          k4a_color_control_mode_t *mode,
                                          int32_t *value);

k4a_result_t k4a_device_set_color_control(k4a_device_t device_handle,
                                          k4a_color_control_command_t command,
                                          k4a_color_control_mode_t mode,
                                          int32_t value);

k4a_buffer_result_t k4a_device_get_raw_calibration(k4a_device_t device_handle,
                                                   uint8_t *data,
                                                   size_t *data_size);

k4a_result_t k4a_device_get_calibration(
        k4a_device_t device_handle,
        const k4a_depth_mode_t depth_mode,
        const k4a_color_resolution_t color_resolution,
        k4a_calibration_t *calibration);

k4a_result_t k4a_device_get_sync_jack(k4a_device_t device_handle,
                                      bool *sync_in_jack_connected,
                                      bool *sync_out_jack_connected);

k4a_result_t k4a_calibration_get_from_raw(
        char *raw_calibration,
        size_t raw_calibration_size,
        const k4a_depth_mode_t depth_mode,
        const k4a_color_resolution_t color_resolution,
        k4a_calibration_t *calibration);

k4a_result_t k4a_calibration_3d_to_3d(
        const k4a_calibration_t *calibration,
        const k4a_float3_t *source_point3d_mm,
        const k4a_calibration_type_t source_camera,
        const k4a_calibration_type_t target_camera,
        k4a_float3_t *target_point3d_mm);

k4a_result_t k4a_calibration_2d_to_3d(
        const k4a_calibration_t *calibration,
        const k4a_float2_t *source_point2d,
        const float source_depth_mm,
        const k4a_calibration_type_t source_camera,
        const k4a_calibration_type_t target_camera,
        k4a_float3_t *target_point3d_mm,
        int *valid);

k4a_result_t k4a_calibration_3d_to_2d(
        const k4a_calibration_t *calibration,
        const k4a_float3_t *source_point3d_mm,
        const k4a_calibration_type_t source_camera,
        const k4a_calibration_type_t target_camera,
        k4a_float2_t *target_point2d,
        int *valid);

k4a_result_t k4a_calibration_2d_to_2d(
        const k4a_calibration_t *calibration,
        const k4a_float2_t *source_point2d,
        const float source_depth_mm,
        const k4a_calibration_type_t source_camera,
        const k4a_calibration_type_t target_camera,
        k4a_float2_t *target_point2d,
        int *valid);

k4a_transformation_t k4a_transformation_create(
        const k4a_calibration_t *calibration);

void k4a_transformation_destroy(k4a_transformation_t transformation_handle);

k4a_result_t k4a_transformation_depth_image_to_color_camera(
        k4a_transformation_t transformation_handle,
        const k4a_image_t depth_image,
        k4a_image_t transformed_depth_image);

k4a_result_t k4a_transformation_color_image_to_depth_camera(
        k4a_transformation_t transformation_handle,
        const k4a_image_t depth_image,
        const k4a_image_t color_image,
        k4a_image_t transformed_color_image);

k4a_result_t k4a_transformation_depth_image_to_point_cloud(
        k4a_transformation_t transformation_handle,
        const k4a_image_t depth_image,
        const k4a_calibration_type_t camera,
        k4a_image_t xyz_image);

}  // namespace k4a_plugin
}  // namespace io
}  // namespace open3d
