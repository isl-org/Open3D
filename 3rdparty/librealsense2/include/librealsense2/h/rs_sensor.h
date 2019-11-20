/* License: Apache 2.0. See LICENSE file in root directory.
   Copyright(c) 2017 Intel Corporation. All Rights Reserved. */

/** \file rs_sensor.h
* \brief
* Exposes RealSense sensor functionality for C compilers
*/


#ifndef LIBREALSENSE_RS2_SENSOR_H
#define LIBREALSENSE_RS2_SENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rs_types.h"

/** \brief Read-only strings that can be queried from the device.
   Not all information attributes are available on all camera types.
   This information is mainly available for camera debug and troubleshooting and should not be used in applications. */
typedef enum rs2_camera_info {
    RS2_CAMERA_INFO_NAME                           , /**< Friendly name */
    RS2_CAMERA_INFO_SERIAL_NUMBER                  , /**< Device serial number */
    RS2_CAMERA_INFO_FIRMWARE_VERSION               , /**< Primary firmware version */
    RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION   , /**< Recommended firmware version */
    RS2_CAMERA_INFO_PHYSICAL_PORT                  , /**< Unique identifier of the port the device is connected to (platform specific) */
    RS2_CAMERA_INFO_DEBUG_OP_CODE                  , /**< If device supports firmware logging, this is the command to send to get logs from firmware */
    RS2_CAMERA_INFO_ADVANCED_MODE                  , /**< True iff the device is in advanced mode */
    RS2_CAMERA_INFO_PRODUCT_ID                     , /**< Product ID as reported in the USB descriptor */
    RS2_CAMERA_INFO_CAMERA_LOCKED                  , /**< True iff EEPROM is locked */
    RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR            , /**< Designated USB specification: USB2/USB3 */
    RS2_CAMERA_INFO_PRODUCT_LINE                   , /**< Device product line D400/SR300/L500/T200 */
    RS2_CAMERA_INFO_ASIC_SERIAL_NUMBER             , /**< ASIC serial number */
    RS2_CAMERA_INFO_FIRMWARE_UPDATE_ID             , /**< Firmware update ID */
    RS2_CAMERA_INFO_COUNT                            /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
} rs2_camera_info;
const char* rs2_camera_info_to_string(rs2_camera_info info);

/** \brief Streams are different types of data provided by RealSense devices. */
typedef enum rs2_stream
{
    RS2_STREAM_ANY,
    RS2_STREAM_DEPTH                            , /**< Native stream of depth data produced by RealSense device */
    RS2_STREAM_COLOR                            , /**< Native stream of color data captured by RealSense device */
    RS2_STREAM_INFRARED                         , /**< Native stream of infrared data captured by RealSense device */
    RS2_STREAM_FISHEYE                          , /**< Native stream of fish-eye (wide) data captured from the dedicate motion camera */
    RS2_STREAM_GYRO                             , /**< Native stream of gyroscope motion data produced by RealSense device */
    RS2_STREAM_ACCEL                            , /**< Native stream of accelerometer motion data produced by RealSense device */
    RS2_STREAM_GPIO                             , /**< Signals from external device connected through GPIO */
    RS2_STREAM_POSE                             , /**< 6 Degrees of Freedom pose data, calculated by RealSense device */
    RS2_STREAM_CONFIDENCE                       , /**< 4 bit per-pixel depth confidence level */
    RS2_STREAM_COUNT
} rs2_stream;
const char* rs2_stream_to_string(rs2_stream stream);

/** \brief A stream's format identifies how binary data is encoded within a frame. */
typedef enum rs2_format
{
    RS2_FORMAT_ANY             , /**< When passed to enable stream, librealsense will try to provide best suited format */
    RS2_FORMAT_Z16             , /**< 16-bit linear depth values. The depth is meters is equal to depth scale * pixel value. */
    RS2_FORMAT_DISPARITY16     , /**< 16-bit float-point disparity values. Depth->Disparity conversion : Disparity = Baseline*FocalLength/Depth. */
    RS2_FORMAT_XYZ32F          , /**< 32-bit floating point 3D coordinates. */
    RS2_FORMAT_YUYV            , /**< 32-bit y0, u, y1, v data for every two pixels. Similar to YUV422 but packed in a different order - https://en.wikipedia.org/wiki/YUV */
    RS2_FORMAT_RGB8            , /**< 8-bit red, green and blue channels */
    RS2_FORMAT_BGR8            , /**< 8-bit blue, green, and red channels -- suitable for OpenCV */
    RS2_FORMAT_RGBA8           , /**< 8-bit red, green and blue channels + constant alpha channel equal to FF */
    RS2_FORMAT_BGRA8           , /**< 8-bit blue, green, and red channels + constant alpha channel equal to FF */
    RS2_FORMAT_Y8              , /**< 8-bit per-pixel grayscale image */
    RS2_FORMAT_Y16             , /**< 16-bit per-pixel grayscale image */
    RS2_FORMAT_RAW10           , /**< Four 10 bits per pixel luminance values packed into a 5-byte macropixel */
    RS2_FORMAT_RAW16           , /**< 16-bit raw image */
    RS2_FORMAT_RAW8            , /**< 8-bit raw image */
    RS2_FORMAT_UYVY            , /**< Similar to the standard YUYV pixel format, but packed in a different order */
    RS2_FORMAT_MOTION_RAW      , /**< Raw data from the motion sensor */
    RS2_FORMAT_MOTION_XYZ32F   , /**< Motion data packed as 3 32-bit float values, for X, Y, and Z axis */
    RS2_FORMAT_GPIO_RAW        , /**< Raw data from the external sensors hooked to one of the GPIO's */
    RS2_FORMAT_6DOF            , /**< Pose data packed as floats array, containing translation vector, rotation quaternion and prediction velocities and accelerations vectors */
    RS2_FORMAT_DISPARITY32     , /**< 32-bit float-point disparity values. Depth->Disparity conversion : Disparity = Baseline*FocalLength/Depth */
    RS2_FORMAT_Y10BPACK        , /**< 16-bit per-pixel grayscale image unpacked from 10 bits per pixel packed ([8:8:8:8:2222]) grey-scale image. The data is unpacked to LSB and padded with 6 zero bits */
    RS2_FORMAT_DISTANCE        , /**< 32-bit float-point depth distance value.  */
    RS2_FORMAT_MJPEG           , /**< Bitstream encoding for video in which an image of each frame is encoded as JPEG-DIB   */
    RS2_FORMAT_COUNT             /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
} rs2_format;
const char* rs2_format_to_string(rs2_format format);

/** \brief Cross-stream extrinsics: encodes the topology describing how the different devices are oriented. */
typedef struct rs2_extrinsics
{
    float rotation[9];    /**< Column-major 3x3 rotation matrix */
    float translation[3]; /**< Three-element translation vector, in meters */
} rs2_extrinsics;

/**
* Deletes sensors list, any sensors created from this list will remain unaffected
* \param[in] info_list list to delete
*/
void rs2_delete_sensor_list(rs2_sensor_list* info_list);

/**
* Determines number of sensors in a list
* \param[in] info_list The list of connected sensors captured using rs2_query_sensors
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            Sensors count
*/
int rs2_get_sensors_count(const rs2_sensor_list* info_list, rs2_error** error);

/**
* delete relasense sensor
* \param[in] sensor realsense sensor to delete
*/
void rs2_delete_sensor(rs2_sensor* sensor);

/**
* create sensor by index
* \param[in] index   the zero based index of sensor to retrieve
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            the requested sensor, should be released by rs2_delete_sensor
*/
rs2_sensor* rs2_create_sensor(const rs2_sensor_list* list, int index, rs2_error** error);

/**
* This is a helper function allowing the user to discover the device from one of its sensors
* \param[in] sensor     Pointer to a sensor
* \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return               new device wrapper for the device of the sensor. Needs to be released by delete_device
*/
rs2_device* rs2_create_device_from_sensor(const rs2_sensor* sensor, rs2_error** error);

/**
* retrieve sensor specific information, like versions of various internal components
* \param[in] sensor     the RealSense sensor
* \param[in] info       camera info type to retrieve
* \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return               the requested camera info string, in a format specific to the device model
*/
const char* rs2_get_sensor_info(const rs2_sensor* sensor, rs2_camera_info info, rs2_error** error);

/**
* check if specific sensor info is supported
* \param[in] info    the parameter to check for support
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return                true if the parameter both exist and well-defined for the specific device
*/
int rs2_supports_sensor_info(const rs2_sensor* sensor, rs2_camera_info info, rs2_error** error);

/**
 * Test if the given sensor can be extended to the requested extension
 * \param[in] sensor  Realsense sensor
 * \param[in] extension The extension to which the sensor should be tested if it is extendable
 * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 * \return non-zero value iff the sensor can be extended to the given extension
 */
int rs2_is_sensor_extendable_to(const rs2_sensor* sensor, rs2_extension extension, rs2_error** error);

/** When called on a depth sensor, this method will return the number of meters represented by a single depth unit
* \param[in] sensor      depth sensor
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return                the number of meters represented by a single depth unit
*/
float rs2_get_depth_scale(rs2_sensor* sensor, rs2_error** error);

/**
* Retrieve the stereoscopic baseline value from frame. Applicable to stereo-based depth modules
* \param[out] float  Stereoscopic baseline in millimeters
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
float rs2_depth_stereo_frame_get_baseline(const rs2_frame* frame_ref, rs2_error** error);

/**
* Retrieve the stereoscopic baseline value from sensor. Applicable to stereo-based depth modules
* \param[out] float  Stereoscopic baseline in millimeters
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
float rs2_get_stereo_baseline(rs2_sensor* sensor, rs2_error** error);

/**
 * \brief sets the active region of interest to be used by auto-exposure algorithm
 * \param[in] sensor     the RealSense sensor
 * \param[in] min_x      lower horizontal bound in pixels
 * \param[in] min_y      lower vertical bound in pixels
 * \param[in] max_x      upper horizontal bound in pixels
 * \param[in] max_y      upper vertical bound in pixels
 * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
void rs2_set_region_of_interest(const rs2_sensor* sensor, int min_x, int min_y, int max_x, int max_y, rs2_error** error);

/**
 * \brief gets the active region of interest to be used by auto-exposure algorithm
 * \param[in] sensor     the RealSense sensor
 * \param[out] min_x     lower horizontal bound in pixels
 * \param[out] min_y     lower vertical bound in pixels
 * \param[out] max_x     upper horizontal bound in pixels
 * \param[out] max_y     upper vertical bound in pixels
 * \param[out] error     if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
void rs2_get_region_of_interest(const rs2_sensor* sensor, int* min_x, int* min_y, int* max_x, int* max_y, rs2_error** error);

/**
* open subdevice for exclusive access, by committing to a configuration
* \param[in] device relevant RealSense device
* \param[in] profile    stream profile that defines single stream configuration
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_open(rs2_sensor* device, const rs2_stream_profile* profile, rs2_error** error);

/**
* open subdevice for exclusive access, by committing to composite configuration, specifying one or more stream profiles
* this method should be used for interdependent  streams, such as depth and infrared, that have to be configured together
* \param[in] device relevant RealSense device
* \param[in] profiles  list of stream profiles discovered by get_stream_profiles
* \param[in] count      number of simultaneous  stream profiles to configure
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_open_multiple(rs2_sensor* device, const rs2_stream_profile** profiles, int count, rs2_error** error);

/**
* stop any streaming from specified subdevice
* \param[in] sensor     RealSense device
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_close(const rs2_sensor* sensor, rs2_error** error);

/**
* start streaming from specified configured sensor
* \param[in] sensor  RealSense device
* \param[in] on_frame function pointer to register as per-frame callback
* \param[in] user auxiliary  data the user wishes to receive together with every frame callback
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_start(const rs2_sensor* sensor, rs2_frame_callback_ptr on_frame, void* user, rs2_error** error);

/**
* start streaming from specified configured sensor
* \param[in] sensor  RealSense device
* \param[in] callback callback object created from c++ application. ownership over the callback object is moved into the relevant streaming lock
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_start_cpp(const rs2_sensor* sensor, rs2_frame_callback* callback, rs2_error** error);

/**
* start streaming from specified configured sensor of specific stream to frame queue
* \param[in] sensor  RealSense Sensor
* \param[in] queue   frame-queue to store new frames into
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_start_queue(const rs2_sensor* sensor, rs2_frame_queue* queue, rs2_error** error);

/**
* stops streaming from specified configured device
* \param[in] sensor  RealSense sensor
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_stop(const rs2_sensor* sensor, rs2_error** error);

/**
* set callback to get notifications from specified sensor
* \param[in] sensor          RealSense device
* \param[in] on_notification function pointer to register as per-notifications callback
* \param[out] error          if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_set_notifications_callback(const rs2_sensor* sensor, rs2_notification_callback_ptr on_notification, void* user, rs2_error** error);

/**
* set callback to get notifications from specified device
* \param[in] sensor  RealSense sensor
* \param[in] callback callback object created from c++ application. ownership over the callback object is moved into the relevant subdevice lock
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_set_notifications_callback_cpp(const rs2_sensor* sensor, rs2_notifications_callback* callback, rs2_error** error);

/**
* retrieve description from notification handle
* \param[in] notification      handle returned from a callback
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            the notification description
*/
const char* rs2_get_notification_description(rs2_notification* notification, rs2_error** error);

/**
* retrieve timestamp from notification handle
* \param[in] notification      handle returned from a callback
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            the notification timestamp
*/
rs2_time_t rs2_get_notification_timestamp(rs2_notification* notification, rs2_error** error);

/**
* retrieve severity from notification handle
* \param[in] notification      handle returned from a callback
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            the notification severity
*/
rs2_log_severity rs2_get_notification_severity(rs2_notification* notification, rs2_error** error);

/**
* retrieve category from notification handle
* \param[in] notification      handle returned from a callback
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            the notification category
*/
rs2_notification_category rs2_get_notification_category(rs2_notification* notification, rs2_error** error);

/**
* retrieve serialized data from notification handle
* \param[in] notification      handle returned from a callback
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            the serialized data (in JSON format)
*/
const char* rs2_get_notification_serialized_data(rs2_notification* notification, rs2_error** error);

/**
* check if physical subdevice is supported
* \param[in] device  input RealSense device
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            list of stream profiles that given subdevice can provide, should be released by rs2_delete_profiles_list
*/
rs2_stream_profile_list* rs2_get_stream_profiles(rs2_sensor* device, rs2_error** error);

/**
* Get pointer to specific stream profile
* \param[in] list        the list of supported profiles returned by rs2_get_supported_profiles
* \param[in] index       the zero based index of the streaming mode
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
const rs2_stream_profile* rs2_get_stream_profile(const rs2_stream_profile_list* list, int index, rs2_error** error);

/**
* Extract common parameters of a stream profiles
* \param[in] mode        input stream profile
* \param[out] stream     stream type of the input profile
* \param[out] format     binary data format of the input profile
* \param[out] index      stream index the input profile in case there are multiple streams of the same type
* \param[out] unique_id  identifier for the stream profile, unique within the application
* \param[out] framerate  expected rate for data frames to arrive, meaning expected number of frames per second
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_get_stream_profile_data(const rs2_stream_profile* mode, rs2_stream* stream, rs2_format* format, int* index, int* unique_id, int* framerate, rs2_error** error);

/**
* Override some of the parameters of the stream profile
* \param[in] mode        input stream profile
* \param[in] stream      stream type for the profile
* \param[in] format      binary data format of the profile
* \param[in] index       stream index the profile in case there are multiple streams of the same type
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_set_stream_profile_data(rs2_stream_profile* mode, rs2_stream stream, int index, rs2_format format, rs2_error** error);

/**
* Creates a copy of stream profile, assigning new values to some of the fields
* \param[in] mode        input stream profile
* \param[in] stream      stream type for the profile
* \param[in] format      binary data format of the profile
* \param[in] index       stream index the profile in case there are multiple streams of the same type
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return                new stream profile, must be deleted by rs2_delete_stream_profile
*/
rs2_stream_profile* rs2_clone_stream_profile(const rs2_stream_profile* mode, rs2_stream stream, int index, rs2_format format, rs2_error** error);

/**
* Creates a copy of stream profile, assigning new values to some of the fields
* \param[in] mode        input stream profile
* \param[in] stream      stream type for the profile
* \param[in] format      binary data format of the profile
* \param[in] width       new width for the profile
* \param[in] height      new height for the profile
* \param[in] intr        new intrinsics for the profile
* \param[in] index       stream index the profile in case there are multiple streams of the same type
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return                new stream profile, must be deleted by rs2_delete_stream_profile
*/
rs2_stream_profile* rs2_clone_video_stream_profile(const rs2_stream_profile* mode, rs2_stream stream, int index, rs2_format format, int width, int height, const rs2_intrinsics* intr, rs2_error** error);


/**
* Delete stream profile allocated by rs2_clone_stream_profile
* Should not be called on stream profiles returned by the device
* \param[in] mode        input stream profile
*/
void rs2_delete_stream_profile(rs2_stream_profile* mode);

/**
* Try to extend stream profile to an extension type
* \param[in] mode        input stream profile
* \param[in] type        extension type, for example RS2_EXTENSION_VIDEO_STREAM_PROFILE
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return                non-zero if profile is extendable to specified extension, zero otherwise
*/
int rs2_stream_profile_is(const rs2_stream_profile* mode, rs2_extension type, rs2_error** error);

/**
* When called on a video stream profile, will return the width and the height of the stream
* \param[in] mode        input stream profile
* \param[out] width      width in pixels of the video stream
* \param[out] height     height in pixels of the video stream
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_get_video_stream_resolution(const rs2_stream_profile* mode, int* width, int* height, rs2_error** error);

/**
* Obtain the intrinsics of a specific stream configuration from the device.
* \param[in] mode          input stream profile
* \param[out] intrinsics   Pointer to the struct to store the data in
* \param[out] error        If non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_get_motion_intrinsics(const rs2_stream_profile* mode, rs2_motion_device_intrinsic * intrinsics, rs2_error ** error);

/**
* Returns non-zero if selected profile is recommended for the sensor
* This is an optional hint we offer to suggest profiles with best performance-quality tradeof
* \param[in] mode        input stream profile
* \param[out] error      if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return                non-zero if selected profile is recommended for the sensor
*/
int rs2_is_stream_profile_default(const rs2_stream_profile* mode, rs2_error** error);

/**
* get the number of supported stream profiles
* \param[in] list        the list of supported profiles returned by rs2_get_supported_profiles
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return number of supported subdevice profiles
*/
int rs2_get_stream_profiles_count(const rs2_stream_profile_list* list, rs2_error** error);

/**
* delete stream profiles list
* \param[in] list        the list of supported profiles returned by rs2_get_supported_profiles
*/
void rs2_delete_stream_profiles_list(rs2_stream_profile_list* list);

/**
 * \param[in] from          origin stream profile
 * \param[in] to            target stream profile
 * \param[out] extrin       extrinsics from origin to target
 * \param[out] error        if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
void rs2_get_extrinsics(const rs2_stream_profile* from,
                        const rs2_stream_profile* to,
                        rs2_extrinsics* extrin, rs2_error** error);

/**
* \param[in] from          origin stream profile
* \param[in] to            target stream profile
* \param[out] extrin       extrinsics from origin to target
* \param[out] error        if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_register_extrinsics(const rs2_stream_profile* from,
    const rs2_stream_profile* to,
    rs2_extrinsics extrin, rs2_error** error);

/**
 * When called on a video profile, returns the intrinsics of specific stream configuration
 * \param[in] mode          input stream profile
 * \param[out] intrinsics   resulting intrinsics for the video profile
 * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
void rs2_get_video_stream_intrinsics(const rs2_stream_profile* mode, rs2_intrinsics* intrinsics, rs2_error** error);

/**
 * Returns the list of recommended processing blocks for a specific sensor. 
 * Order and configuration of the blocks are decided by the sensor
 * \param[in] sensor          input sensor
 * \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
 * \return list of supported sensor recommended processing blocks
*/
rs2_processing_block_list* rs2_get_recommended_processing_blocks(rs2_sensor* sensor, rs2_error** error);

/**
* Returns specific processing blocks from processing blocks list
* \param[in] list           the processing blocks list
* \param[in] index          the requested processing block
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return processing block
*/
rs2_processing_block* rs2_get_processing_block(const rs2_processing_block_list* list, int index, rs2_error** error);

/**
* Returns the processing blocks list size
* \param[in] list           the processing blocks list
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return the processing block list size
*/
int rs2_get_recommended_processing_blocks_count(const rs2_processing_block_list* list, rs2_error** error);

/**
* Deletes processing blocks list
* \param[in] list list to delete
*/
void rs2_delete_recommended_processing_blocks(rs2_processing_block_list* list);

/**
* Imports a localization map from file to tm2 tracking device
* \param[in]  sensor        TM2 position-tracking sensor
* \param[in]  lmap_blob     Localization map raw buffer, serialized
* \param[in]  blob_size     The buffer's size in bytes
* \param[out] error         If non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return                   Non-zero if succeeded, otherwise 0
*/
int rs2_import_localization_map(const rs2_sensor* sensor, const unsigned char* lmap_blob, unsigned int blob_size, rs2_error** error);

/**
* Extract and store the localization map of tm2 tracking device to file
* \param[in]  sensor        TM2 position-tracking sensor
* \param[in]  lmap_fname    The file name of the localization map
* \param[out] error         If non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return                   Device's response in a rs2_raw_data_buffer, which should be released by rs2_delete_raw_data
*/
//void rs2_export_localization_map(const rs2_sensor* sensor, const char* lmap_fname, rs2_error** error);
const rs2_raw_data_buffer* rs2_export_localization_map(const rs2_sensor* sensor, rs2_error** error);

/**
* Create a named location tag
* \param[in]  sensor    T2xx position-tracking sensor
* \param[in]  guid      Null-terminated string of up to 127 characters
* \param[in]  pos       Position in meters, relative to the current tracking session
* \param[in]  orient    Quaternion orientation, expressed the the coordinate system of the current tracking session
* \param[out] error     If non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return               Non-zero if succeeded, otherwise 0
*/
int rs2_set_static_node(const rs2_sensor* sensor, const char* guid, const rs2_vector pos, const rs2_quaternion orient, rs2_error** error);

/**
* Create a named location tag
* \param[in]  sensor    T2xx position-tracking sensor
* \param[in]  guid      Null-terminated string of up to 127 characters
* \param[out] pos       Position in meters of the tagged (stored) location
* \param[out] orient    Quaternion orientation of the tagged (stored) location
* \param[out] error     If non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return               Non-zero if succeeded, otherwise 0
*/
int rs2_get_static_node(const rs2_sensor* sensor, const char* guid, rs2_vector *pos, rs2_quaternion *orient, rs2_error** error);

/** Load Wheel odometer settings from host to device
* \param[in] odometry_config_buf   odometer configuration/calibration blob serialized from jsom file
* \return true on success
*/
int rs2_load_wheel_odometry_config(const rs2_sensor* sensor, const unsigned char* odometry_config_buf, unsigned int blob_size, rs2_error** error);

/** Send wheel odometry data for each individual sensor (wheel)
* \param[in] wo_sensor_id       - Zero-based index of (wheel) sensor with the same type within device
* \param[in] frame_num          - Monotonocally increasing frame number, managed per sensor.
* \param[in] translational_velocity   - Translational velocity of the wheel sensor [meter/sec]
* \return true on success
*/
int rs2_send_wheel_odometry(const rs2_sensor* sensor, char wo_sensor_id, unsigned int frame_num,
    const rs2_vector translational_velocity, rs2_error** error);

/**
* Set intrinsics of a given sensor
* \param[in] sensor       The RealSense device
* \param[in] profile      Target stream profile
* \param[in] intrinsics   Intrinsics value to be written to the device
* \param[out] error       If non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_set_intrinsics(const rs2_sensor* sensor, const rs2_stream_profile* profile , const rs2_intrinsics* intrinsics, rs2_error** error);

/**
 * Set extrinsics between two sensors
 * \param[in]  from_sensor  Origin sensor
 * \param[in]  from_profile Origin profile
 * \param[in]  to_sensor    Target sensor
 * \param[in]  to_profile   Target profile
 * \param[out] extrinsics   Extrinsics from origin to target
 * \param[out] error        If non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
void rs2_set_extrinsics(const rs2_sensor* from_sensor, const rs2_stream_profile* from_profile, rs2_sensor* to_sensor, const rs2_stream_profile* to_profile, const rs2_extrinsics* extrinsics, rs2_error** error);

/**
* Set motion device intrinsics
* \param[in]  sensor       Motion sensor 
* \param[in]  profile      Motion stream profile
* \param[out] intrinsics   Pointer to the struct to store the data in
* \param[out] error        If non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_set_motion_device_intrinsics(const rs2_sensor* sensor, const rs2_stream_profile* profile, const rs2_motion_device_intrinsic* intrinsics, rs2_error** error);


#ifdef __cplusplus
}
#endif
#endif  // LIBREALSENSE_RS2_SENSOR_H
