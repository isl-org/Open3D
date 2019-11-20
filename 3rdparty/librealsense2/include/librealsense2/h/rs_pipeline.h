/* License: Apache 2.0. See LICENSE file in root directory.
Copyright(c) 2017 Intel Corporation. All Rights Reserved. */

/** \file rs_pipeline.h
* \brief
* Exposes RealSense processing-block functionality for C compilers
*/


#ifndef LIBREALSENSE_RS2_PIPELINE_H
#define LIBREALSENSE_RS2_PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rs_types.h"
#include "rs_sensor.h"
#include "rs_config.h"

    /**
    * Create a pipeline instance
    * The pipeline simplifies the user interaction with the device and computer vision processing modules.
    * The class abstracts the camera configuration and streaming, and the vision modules triggering and threading.
    * It lets the application focus on the computer vision output of the modules, or the device output data.
    * The pipeline can manage computer vision modules, which are implemented as a processing blocks.
    * The pipeline is the consumer of the processing block interface, while the application consumes the
    * computer vision interface.
    * \param[in]  ctx    context
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    rs2_pipeline* rs2_create_pipeline(rs2_context* ctx, rs2_error ** error);

    /**
    * Stop the pipeline streaming.
    * The pipeline stops delivering samples to the attached computer vision modules and processing blocks, stops the device streaming
    * and releases the device resources used by the pipeline. It is the application's responsibility to release any frame reference it owns.
    * The method takes effect only after \c start() was called, otherwise an exception is raised.
    * \param[in] pipe  pipeline
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    */
    void rs2_pipeline_stop(rs2_pipeline* pipe, rs2_error ** error);

    /**
    * Wait until a new set of frames becomes available.
    * The frames set includes time-synchronized frames of each enabled stream in the pipeline.
    * The method blocks the calling thread, and fetches the latest unread frames set.
    * Device frames, which were produced while the function wasn't called, are dropped. To avoid frame drops, this method should be called
    * as fast as the device frame rate.
    * The application can maintain the frames handles to defer processing. However, if the application maintains too long history, the device
    * may lack memory resources to produce new frames, and the following call to this method shall fail to retrieve new frames, until resources
    * are retained.
    * \param[in] pipe the pipeline
    * \param[in] timeout_ms   Max time in milliseconds to wait until an exception will be thrown
    * \param[out] error         if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return Set of coherent frames
    */
    rs2_frame* rs2_pipeline_wait_for_frames(rs2_pipeline* pipe, unsigned int timeout_ms, rs2_error ** error);

    /**
    * Check if a new set of frames is available and retrieve the latest undelivered set.
    * The frames set includes time-synchronized frames of each enabled stream in the pipeline.
    * The method returns without blocking the calling thread, with status of new frames available or not. If available, it fetches the
    * latest frames set.
    * Device frames, which were produced while the function wasn't called, are dropped. To avoid frame drops, this method should be called
    * as fast as the device frame rate.
    * The application can maintain the frames handles to defer processing. However, if the application maintains too long history, the device
    * may lack memory resources to produce new frames, and the following calls to this method shall return no new frames, until resources are
    * retained.
    * \param[in] pipe the pipeline
    * \param[out] output_frame frame handle to be released using rs2_release_frame
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return true if new frame was stored to output_frame
    */
    int rs2_pipeline_poll_for_frames(rs2_pipeline* pipe, rs2_frame** output_frame, rs2_error ** error);

    /**
    * Wait until a new set of frames becomes available.
    * The frames set includes time-synchronized frames of each enabled stream in the pipeline.
    * The method blocks the calling thread, and fetches the latest unread frames set.
    * Device frames, which were produced while the function wasn't called, are dropped. To avoid frame drops, this method should be called
    * as fast as the device frame rate.
    * The application can maintain the frames handles to defer processing. However, if the application maintains too long history, the device
    * may lack memory resources to produce new frames, and the following call to this method shall fail to retrieve new frames, until resources
    * are retained.
    * \param[in] pipe           the pipeline
    * \param[in] timeout_ms     max time in milliseconds to wait until a frame becomes available
    * \param[out] output_frame  frame handle to be released using rs2_release_frame
    * \param[out] error         if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return true if new frame was stored to output_frame
    */
    int rs2_pipeline_try_wait_for_frames(rs2_pipeline* pipe, rs2_frame** output_frame, unsigned int timeout_ms, rs2_error ** error);

    /**
    * Delete a pipeline instance.
    * Upon destruction, the pipeline will implicitly stop itself
    * \param[in] pipe to delete
    */
    void rs2_delete_pipeline(rs2_pipeline* pipe);

    /**
    * Start the pipeline streaming with its default configuration.
    * The pipeline streaming loop captures samples from the device, and delivers them to the attached computer vision modules
    * and processing blocks, according to each module requirements and threading model.
    * During the loop execution, the application can access the camera streams by calling \c wait_for_frames() or \c poll_for_frames().
    * The streaming loop runs until the pipeline is stopped.
    * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
    *
    * \param[in] pipe    a pointer to an instance of the pipeline
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return             The actual pipeline device and streams profile, which was successfully configured to the streaming device.
    */
    rs2_pipeline_profile* rs2_pipeline_start(rs2_pipeline* pipe, rs2_error ** error);

    /**
    * Start the pipeline streaming according to the configuraion.
    * The pipeline streaming loop captures samples from the device, and delivers them to the attached computer vision modules
    * and processing blocks, according to each module requirements and threading model.
    * During the loop execution, the application can access the camera streams by calling \c wait_for_frames() or \c poll_for_frames().
    * The streaming loop runs until the pipeline is stopped.
    * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
    * The pipeline selects and activates the device upon start, according to configuration or a default configuration.
    * When the rs2::config is provided to the method, the pipeline tries to activate the config \c resolve() result. If the application
    * requests are conflicting with pipeline computer vision modules or no matching device is available on the platform, the method fails.
    * Available configurations and devices may change between config \c resolve() call and pipeline start, in case devices are connected
    * or disconnected, or another application acquires ownership of a device.
    *
    * \param[in] pipe    a pointer to an instance of the pipeline
    * \param[in] config   A rs2::config with requested filters on the pipeline configuration. By default no filters are applied.
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return             The actual pipeline device and streams profile, which was successfully configured to the streaming device.
    */
    rs2_pipeline_profile* rs2_pipeline_start_with_config(rs2_pipeline* pipe, rs2_config* config, rs2_error ** error);

    /**
    * Start the pipeline streaming with its default configuration.
    * The pipeline captures samples from the device, and delivers them to the through the provided frame callback.
    * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
    * When starting the pipeline with a callback both \c wait_for_frames() or \c poll_for_frames() will throw exception.
    *
    * \param[in] pipe     A pointer to an instance of the pipeline
    * \param[in] on_frame function pointer to register as per-frame callback
    * \param[in] user auxiliary  data the user wishes to receive together with every frame callback
    * \param[out] error   If non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return             The actual pipeline device and streams profile, which was successfully configured to the streaming device.
    */
    rs2_pipeline_profile* rs2_pipeline_start_with_callback(rs2_pipeline* pipe, rs2_frame_callback_ptr on_frame, void* user, rs2_error ** error);

    /**
    * Start the pipeline streaming with its default configuration.
    * The pipeline captures samples from the device, and delivers them to the through the provided frame callback.
    * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
    * When starting the pipeline with a callback both \c wait_for_frames() or \c poll_for_frames() will throw exception. 
    *
    * \param[in] pipe     A pointer to an instance of the pipeline
    * \param[in] callback callback object created from c++ application. ownership over the callback object is moved into the relevant streaming lock
    * \param[out] error   If non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return             The actual pipeline device and streams profile, which was successfully configured to the streaming device.
    */
    rs2_pipeline_profile* rs2_pipeline_start_with_callback_cpp(rs2_pipeline* pipe, rs2_frame_callback* callback, rs2_error ** error);

    /**
    * Start the pipeline streaming according to the configuraion.
    * The pipeline captures samples from the device, and delivers them to the through the provided frame callback.
    * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
    * When starting the pipeline with a callback both \c wait_for_frames() or \c poll_for_frames() will throw exception.
    * The pipeline selects and activates the device upon start, according to configuration or a default configuration.
    * When the rs2::config is provided to the method, the pipeline tries to activate the config \c resolve() result. If the application
    * requests are conflicting with pipeline computer vision modules or no matching device is available on the platform, the method fails.
    * Available configurations and devices may change between config \c resolve() call and pipeline start, in case devices are connected
    * or disconnected, or another application acquires ownership of a device.
    *
    * \param[in] pipe     A pointer to an instance of the pipeline
    * \param[in] config   A rs2::config with requested filters on the pipeline configuration. By default no filters are applied.
    * \param[in] on_frame function pointer to register as per-frame callback
    * \param[in] user auxiliary  data the user wishes to receive together with every frame callback
    * \param[out] error   If non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return             The actual pipeline device and streams profile, which was successfully configured to the streaming device.
    */
    rs2_pipeline_profile* rs2_pipeline_start_with_config_and_callback(rs2_pipeline* pipe, rs2_config* config, rs2_frame_callback_ptr on_frame, void* user, rs2_error ** error);

    /**
    * Start the pipeline streaming according to the configuraion.
    * The pipeline captures samples from the device, and delivers them to the through the provided frame callback.
    * Starting the pipeline is possible only when it is not started. If the pipeline was started, an exception is raised.
    * When starting the pipeline with a callback both \c wait_for_frames() or \c poll_for_frames() will throw exception.
    * The pipeline selects and activates the device upon start, according to configuration or a default configuration.
    * When the rs2::config is provided to the method, the pipeline tries to activate the config \c resolve() result. If the application
    * requests are conflicting with pipeline computer vision modules or no matching device is available on the platform, the method fails.
    * Available configurations and devices may change between config \c resolve() call and pipeline start, in case devices are connected
    * or disconnected, or another application acquires ownership of a device.
    *
    * \param[in] pipe     A pointer to an instance of the pipeline
    * \param[in] config   A rs2::config with requested filters on the pipeline configuration. By default no filters are applied.
    * \param[in] callback callback object created from c++ application. ownership over the callback object is moved into the relevant streaming lock
    * \param[out] error   If non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return             The actual pipeline device and streams profile, which was successfully configured to the streaming device.
    */
    rs2_pipeline_profile* rs2_pipeline_start_with_config_and_callback_cpp(rs2_pipeline* pipe, rs2_config* config, rs2_frame_callback* callback, rs2_error ** error);

    /**
    * Return the active device and streams profiles, used by the pipeline.
    * The pipeline streams profiles are selected during \c start(). The method returns a valid result only when the pipeline is active -
    * between calls to \c start() and \c stop().
    * After \c stop() is called, the pipeline doesn't own the device, thus, the pipeline selected device may change in subsequent activations.
    *
    * \param[in] pipe    a pointer to an instance of the pipeline
    * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return  The actual pipeline device and streams profile, which was successfully configured to the streaming device on start.
    */
    rs2_pipeline_profile* rs2_pipeline_get_active_profile(rs2_pipeline* pipe, rs2_error ** error);

    /**
    * Retrieve the device used by the pipeline.
    * The device class provides the application access to control camera additional settings -
    * get device information, sensor options information, options value query and set, sensor specific extensions.
    * Since the pipeline controls the device streams configuration, activation state and frames reading, calling
    * the device API functions, which execute those operations, results in unexpected behavior.
    * The pipeline streaming device is selected during pipeline \c start(). Devices of profiles, which are not returned by
    * pipeline \c start() or \c get_active_profile(), are not guaranteed to be used by the pipeline.
    *
    * \param[in] profile    A pointer to an instance of a pipeline profile
    * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return rs2_device* The pipeline selected device
    */
    rs2_device* rs2_pipeline_profile_get_device(rs2_pipeline_profile* profile, rs2_error ** error);

    /**
    * Return the selected streams profiles, which are enabled in this profile.
    *
    * \param[in] profile    A pointer to an instance of a pipeline profile
    * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
    * \return   list of stream profiles
    */
    rs2_stream_profile_list* rs2_pipeline_profile_get_streams(rs2_pipeline_profile* profile, rs2_error** error);

    /**
    * Deletes an instance of a pipeline profile
    *
    * \param[in] profile    A pointer to an instance of a pipeline profile
    */
    void rs2_delete_pipeline_profile(rs2_pipeline_profile* profile);

#ifdef __cplusplus
}
#endif
#endif
