/* License: Apache 2.0. See LICENSE file in root directory.
   Copyright(c) 2017 Intel Corporation. All Rights Reserved. */

/** \file rs_processing.h
* \brief
* Exposes RealSense processing-block functionality for C compilers
*/


#ifndef LIBREALSENSE_RS2_PROCESSING_H
#define LIBREALSENSE_RS2_PROCESSING_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rs_types.h"
#include "rs_sensor.h"
#include "rs_option.h"

/**
* Creates Depth-Colorizer processing block that can be used to quickly visualize the depth data
* This block will accept depth frames as input and replace them by depth frames with format RGB8
* Non-depth frames are passed through
* Further customization will be added soon (format, color-map, histogram equalization control)
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_colorizer(rs2_error** error);

/**
* Creates Sync processing block. This block accepts arbitrary frames and output composite frames of best matches
* Some frames may be released within the syncer if they are waiting for match for too long
* Syncronization is done (mostly) based on timestamps so good hardware timestamps are a pre-condition
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_sync_processing_block(rs2_error** error);

/**
* Creates Point-Cloud processing block. This block accepts depth frames and outputs Points frames
* In addition, given non-depth frame, the block will align texture coordinate to the non-depth stream
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_pointcloud(rs2_error** error);

/**
* Creates YUY decoder processing block. This block accepts raw YUY frames and outputs frames of other formats.
* YUY is a common video format used by a variety of web-cams. It benefits from packing pixels into 2 bytes per pixel
* without signficant quality drop. YUY representation can be converted back to more usable RGB form,
* but this requires somewhat costly conversion.
* The SDK will automatically try to use SSE2 and AVX instructions and CUDA where available to get
* best performance. Other implementations (using GLSL, OpenCL, Neon and NCS) should follow.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_yuy_decoder(rs2_error** error);

/**
* Creates depth thresholding processing block
* By controlling min and max options on the block, one could filter out depth values
* that are either too large or too small, as a software post-processing step
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_threshold(rs2_error** error);

/**
* Creates depth units transformation processing block
* All of the pixels are transformed from depth units into meters.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_units_transform(rs2_error** error);

/**
* This method creates new custom processing block. This lets the users pass frames between module boundaries for processing
* This is an infrastructure function aimed at middleware developers, and also used by provided blocks such as sync, colorizer, etc..
* \param proc       Processing function to be applied to every frame entering the block
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return           new processing block, to be released by rs2_delete_processing_block
*/
rs2_processing_block* rs2_create_processing_block(rs2_frame_processor_callback* proc, rs2_error** error);

/**
* This method creates new custom processing block from function pointer. This lets the users pass frames between module boundaries for processing
* This is an infrastructure function aimed at middleware developers, and also used by provided blocks such as sync, colorizer, etc..
* \param proc       Processing function pointer to be applied to every frame entering the block
* \param context    User context (can be anything or null) to be passed later as ctx param of the callback
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return           new processing block, to be released by rs2_delete_processing_block
*/
rs2_processing_block* rs2_create_processing_block_fptr(rs2_frame_processor_callback_ptr proc, void * context, rs2_error** error);

/**
* This method adds a custom option to a custom processing block. This is a simple float that can be accessed via rs2_set_option and rs2_get_option
* This is an infrastructure function aimed at middleware developers, and also used by provided blocks such as save_to_ply, etc..
* \param[in] block      Processing block
* \param[in] option_id  an int ID for referencing the option
* \param[in] min     the minimum value which will be accepted for this option
* \param[in] max     the maximum value which will be accepted for this option
* \param[in] step    the granularity of options which accept discrete values, or zero if the option accepts continuous values
* \param[in] def     the default value of the option. This will be the initial value.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            true if adding the option succeeds. false if it fails e.g. an option with this id is already registered
*/
int rs2_processing_block_register_simple_option(rs2_processing_block* block, rs2_option option_id, float min, float max, float step, float def, rs2_error** error);

/**
* This method is used to direct the output from the processing block to some callback or sink object
* \param[in] block          Processing block
* \param[in] on_frame       Callback to be invoked every time the processing block calls frame_ready
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_start_processing(rs2_processing_block* block, rs2_frame_callback* on_frame, rs2_error** error);

/**
* This method is used to direct the output from the processing block to some callback or sink object
* \param[in] block          Processing block
* \param[in] on_frame       Callback function to be invoked every time the processing block calls frame_ready
* \param[in] user           User context for the callback (can be anything or null)
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_start_processing_fptr(rs2_processing_block* block, rs2_frame_callback_ptr on_frame, void* user, rs2_error** error);

/**
* This method is used to direct the output from the processing block to a dedicated queue object
* \param[in] block          Processing block
* \param[in] queue          Queue to place the processed frames to
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_start_processing_queue(rs2_processing_block* block, rs2_frame_queue* queue, rs2_error** error);

/**
* This method is used to pass frame into a processing block
* \param[in] block          Processing block
* \param[in] frame          Frame to process, ownership is moved to the block object
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_process_frame(rs2_processing_block* block, rs2_frame* frame, rs2_error** error);

/**
* Deletes the processing block
* \param[in] block          Processing block
*/
void rs2_delete_processing_block(rs2_processing_block* block);

/**
* create frame queue. frame queues are the simplest x-platform synchronization primitive provided by librealsense
* to help developers who are not using async APIs
* \param[in] capacity max number of frames to allow to be stored in the queue before older frames will start to get dropped
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return handle to the frame queue, must be released using rs2_delete_frame_queue
*/
rs2_frame_queue* rs2_create_frame_queue(int capacity, rs2_error** error);

/**
* deletes frame queue and releases all frames inside it
* \param[in] queue queue to delete
*/
void rs2_delete_frame_queue(rs2_frame_queue* queue);

/**
* wait until new frame becomes available in the queue and dequeue it
* \param[in] queue the frame queue data structure
* \param[in] timeout_ms   max time in milliseconds to wait until an exception will be thrown
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return frame handle to be released using rs2_release_frame
*/
rs2_frame* rs2_wait_for_frame(rs2_frame_queue* queue, unsigned int timeout_ms, rs2_error** error);

/**
* poll if a new frame is available and dequeue if it is
* \param[in] queue the frame queue data structure
* \param[out] output_frame frame handle to be released using rs2_release_frame
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return true if new frame was stored to output_frame
*/
int rs2_poll_for_frame(rs2_frame_queue* queue, rs2_frame** output_frame, rs2_error** error);

/**
* wait until new frame becomes available in the queue and dequeue it
* \param[in] queue          the frame queue data structure
* \param[in] timeout_ms     max time in milliseconds to wait until a frame becomes available
* \param[out] output_frame  frame handle to be released using rs2_release_frame
* \param[out] error         if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return true if new frame was stored to output_frame
*/
int rs2_try_wait_for_frame(rs2_frame_queue* queue, unsigned int timeout_ms, rs2_frame** output_frame, rs2_error** error);

/**
* enqueue new frame into a queue
* \param[in] frame frame handle to enqueue (this operation passed ownership to the queue)
* \param[in] queue the frame queue data structure
*/
void rs2_enqueue_frame(rs2_frame* frame, void* queue);

/**
* Creates Align processing block.
* \param[in] align_to   stream type to be used as the target of frameset alignment
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_align(rs2_stream align_to, rs2_error** error);

/**
* Creates Depth post-processing filter block. This block accepts depth frames, applies decimation filter and plots modified prames
* Note that due to the modifiedframe size, the decimated frame repaces the original one
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_decimation_filter_block(rs2_error** error);

/**
* Creates Depth post-processing filter block. This block accepts depth frames, applies temporal filter
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_temporal_filter_block(rs2_error** error);

/**
* Creates Depth post-processing spatial filter block. This block accepts depth frames, applies spatial filters and plots modified prames
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_spatial_filter_block(rs2_error** error);

/**
* Creates a post processing block that provides for depth<->disparity domain transformation for stereo-based depth modules
* \param[in] transform_to_disparity flag select the transform direction:  true = depth->disparity, and vice versa
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_disparity_transform_block(unsigned char transform_to_disparity, rs2_error** error);

/**
* Creates Depth post-processing hole filling block. The filter replaces empty pixels with data from adjacent pixels based on the method selected
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_hole_filling_filter_block(rs2_error** error);

/**
* Creates a rates printer block. The printer prints the actual FPS of the invoked frame stream.
* The block ignores reapiting frames and calculats the FPS only if the frame number of the relevant frame was changed.
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
rs2_processing_block* rs2_create_rates_printer_block(rs2_error** error);

/**
* Creates Depth post-processing zero order fix block. The filter invalidates pixels that has a wrong value due to zero order effect
* \param[out] error     If non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return               zero order fix processing block
*/
rs2_processing_block* rs2_create_zero_order_invalidation_block(rs2_error** error);

/**
* Retrieve processing block specific information, like name.
* \param[in]  block     The processing block
* \param[in]  info      processing block info type to retrieve
* \param[out] error     If non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return               The requested processing block info string, in a format specific to the device model
*/
const char* rs2_get_processing_block_info(const rs2_processing_block* block, rs2_camera_info info, rs2_error** error);

/**
* Check if a processing block supports a specific info type.
* \param[in]  block     The processing block to check
* \param[in]  info      The parameter to check for support
* \param[out] error     If non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return               True if the parameter both exist and well-defined for the specific device
*/
int rs2_supports_processing_block_info(const rs2_processing_block* block, rs2_camera_info info, rs2_error** error);

/**
 * Test if the given processing block can be extended to the requested extension
 * \param[in] block processing block
 * \param[in] extension The extension to which the sensor should be tested if it is extendable
 * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 * \return non-zero value iff the processing block can be extended to the given extension
 */
int rs2_is_processing_block_extendable_to(const rs2_processing_block* block, rs2_extension extension_type, rs2_error** error);

#ifdef __cplusplus
}
#endif
#endif
