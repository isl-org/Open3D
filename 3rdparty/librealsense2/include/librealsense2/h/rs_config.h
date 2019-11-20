/* License: Apache 2.0. See LICENSE file in root directory.
Copyright(c) 2017 Intel Corporation. All Rights Reserved. */

/** \file rs_pipeline.h
* \brief
* Exposes RealSense processing-block functionality for C compilers
*/


#ifndef LIBREALSENSE_RS2_CONFIG_H
#define LIBREALSENSE_RS2_CONFIG_H

#define RS2_DEFAULT_TIMEOUT 15000

#ifdef __cplusplus
extern "C" {
#endif

#include "rs_types.h"
#include "rs_sensor.h"

    /**
    * Create a config instance
    * The config allows pipeline users to request filters for the pipeline streams and device selection and configuration.
    * This is an optional step in pipeline creation, as the pipeline resolves its streaming device internally.
    * Config provides its users a way to set the filters and test if there is no conflict with the pipeline requirements
    * from the device. It also allows the user to find a matching device for the config filters and the pipeline, in order to
    * select a device explicitly, and modify its controls before streaming starts.
    *
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return rs2_config*  A pointer to a new config instance
    */
    rs2_config* rs2_create_config(rs2_error** error);

    /**
    * Deletes an instance of a config
    *
    * \param[in] config    A pointer to an instance of a config
    */
    void rs2_delete_config(rs2_config* config);

    /**
    * Enable a device stream explicitly, with selected stream parameters.
    * The method allows the application to request a stream with specific configuration. If no stream is explicitly enabled, the pipeline
    * configures the device and its streams according to the attached computer vision modules and processing blocks requirements, or
    * default configuration for the first available device.
    * The application can configure any of the input stream parameters according to its requirement, or set to 0 for don't care value.
    * The config accumulates the application calls for enable configuration methods, until the configuration is applied. Multiple enable
    * stream calls for the same stream with conflicting parameters override each other, and the last call is maintained.
    * Upon calling \c resolve(), the config checks for conflicts between the application configuration requests and the attached computer
    * vision modules and processing blocks requirements, and fails if conflicts are found. Before \c resolve() is called, no conflict
    * check is done.
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[in] stream    Stream type to be enabled
    * \param[in] index     Stream index, used for multiple streams of the same type. -1 indicates any.
    * \param[in] width     Stream image width - for images streams. 0 indicates any.
    * \param[in] height    Stream image height - for images streams. 0 indicates any.
    * \param[in] format    Stream data format - pixel format for images streams, of data type for other streams. RS2_FORMAT_ANY indicates any.
    * \param[in] framerate Stream frames per second. 0 indicates any.
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_enable_stream(rs2_config* config,
        rs2_stream stream,
        int index,
        int width,
        int height,
        rs2_format format,
        int framerate,
        rs2_error** error);

    /**
    * Enable all device streams explicitly.
    * The conditions and behavior of this method are similar to those of \c enable_stream().
    * This filter enables all raw streams of the selected device. The device is either selected explicitly by the application,
    * or by the pipeline requirements or default. The list of streams is device dependent.
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_enable_all_stream(rs2_config* config, rs2_error ** error);

    /**
    * Select a specific device explicitly by its serial number, to be used by the pipeline.
    * The conditions and behavior of this method are similar to those of \c enable_stream().
    * This method is required if the application needs to set device or sensor settings prior to pipeline streaming, to enforce
    * the pipeline to use the configured device.
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[in] serial device serial number, as returned by RS2_CAMERA_INFO_SERIAL_NUMBER
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_enable_device(rs2_config* config, const char* serial, rs2_error ** error);

    /**
    * Select a recorded device from a file, to be used by the pipeline through playback.
    * The device available streams are as recorded to the file, and \c resolve() considers only this device and configuration
    * as available.
    * This request cannot be used if enable_record_to_file() is called for the current config, and vise versa
    * By default, playback is repeated once the file ends. To control this, see 'rs2_config_enable_device_from_file_repeat_option'.
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[in] file      The playback file of the device
    * \param[out] error    if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_enable_device_from_file(rs2_config* config, const char* file, rs2_error ** error);

    /**
    * Select a recorded device from a file, to be used by the pipeline through playback.
    * The device available streams are as recorded to the file, and \c resolve() considers only this device and configuration
    * as available.
    * This request cannot be used if enable_record_to_file() is called for the current config, and vise versa
    *
    * \param[in] config           A pointer to an instance of a config
    * \param[in] file             The playback file of the device
    * \param[in] repeat_playback  if true, when file ends the playback starts again, in an infinite loop;
                                  if false, when file ends playback does not start again, and should by stopped manually by the user.
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_enable_device_from_file_repeat_option(rs2_config* config, const char* file, int repeat_playback, rs2_error ** error);

    /**
    * Requires that the resolved device would be recorded to file
    * This request cannot be used if enable_device_from_file() is called for the current config, and vise versa
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[in] file      The desired file for the output record
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_enable_record_to_file(rs2_config* config, const char* file, rs2_error ** error);


    /**
    * Disable a device stream explicitly, to remove any requests on this stream type.
    * The stream can still be enabled due to pipeline computer vision module request. This call removes any filter on the
    * stream configuration.
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[in] stream    Stream type, for which the filters are cleared
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_disable_stream(rs2_config* config, rs2_stream stream, rs2_error ** error);

    /**
    * Disable a device stream explicitly, to remove any requests on this stream profile.
    * The stream can still be enabled due to pipeline computer vision module request. This call removes any filter on the
    * stream configuration.
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[in] stream    Stream type, for which the filters are cleared
    * \param[in] index     Stream index, for which the filters are cleared
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_disable_indexed_stream(rs2_config* config, rs2_stream stream, int index, rs2_error ** error);

    /**
    * Disable all device stream explicitly, to remove any requests on the streams profiles.
    * The streams can still be enabled due to pipeline computer vision module request. This call removes any filter on the
    * streams configuration.
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_config_disable_all_streams(rs2_config* config, rs2_error ** error);

    /**
    * Resolve the configuration filters, to find a matching device and streams profiles.
    * The method resolves the user configuration filters for the device and streams, and combines them with the requirements of
    * the computer vision modules and processing blocks attached to the pipeline. If there are no conflicts of requests, it looks
    * for an available device, which can satisfy all requests, and selects the first matching streams configuration. In the absence
    * of any request, the rs2::config selects the first available device and the first color and depth streams configuration.
    * The pipeline profile selection during \c start() follows the same method. Thus, the selected profile is the same, if no
    * change occurs to the available devices occurs.
    * Resolving the pipeline configuration provides the application access to the pipeline selected device for advanced control.
    * The returned configuration is not applied to the device, so the application doesn't own the device sensors. However, the
    * application can call \c enable_device(), to enforce the device returned by this method is selected by pipeline \c start(),
    * and configure the device and sensors options or extensions before streaming starts.
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[in] pipe  The pipeline for which the selected filters are applied
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return       A matching device and streams profile, which satisfies the filters and pipeline requests.
    */
    rs2_pipeline_profile* rs2_config_resolve(rs2_config* config, rs2_pipeline* pipe, rs2_error ** error);

    /**
    * Check if the config can resolve the configuration filters, to find a matching device and streams profiles.
    * The resolution conditions are as described in \c resolve().
    *
    * \param[in] config    A pointer to an instance of a config
    * \param[in] pipe  The pipeline for which the selected filters are applied
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return       True if a valid profile selection exists, false if no selection can be found under the config filters and the available devices.
    */
    int rs2_config_can_resolve(rs2_config* config, rs2_pipeline* pipe, rs2_error ** error);

#ifdef __cplusplus
}
#endif
#endif
