/* License: Apache 2.0. See LICENSE file in root directory.
   Copyright(c) 2017 Intel Corporation. All Rights Reserved. */

/** \file rs_context.h
* \brief Exposes RealSense context functionality for C compilers
*/


#ifndef LIBREALSENSE_RS2_CONTEXT_H
#define LIBREALSENSE_RS2_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif
#include "rs_types.h"

/**
* \brief Creates RealSense context that is required for the rest of the API.
* \param[in] api_version Users are expected to pass their version of \c RS2_API_VERSION to make sure they are running the correct librealsense version.
* \param[out] error  If non-null, receives any error that occurs during this call, otherwise, errors are ignored.
* \return            Context object
*/
rs2_context* rs2_create_context(int api_version, rs2_error** error);

/**
* \brief Frees the relevant context object.
* \param[in] context Object that is no longer needed
*/
void rs2_delete_context(rs2_context* context);

/**
* set callback to get devices changed events
* these events will be raised by the context whenever new RealSense device is connected or existing device gets disconnected
* \param context     Object representing librealsense session
* \param[in] callback callback object created from c++ application. ownership over the callback object is moved into the context
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_set_devices_changed_callback_cpp(rs2_context* context, rs2_devices_changed_callback* callback, rs2_error** error);

/**
* set callback to get devices changed events
* these events will be raised by the context whenever new RealSense device is connected or existing device gets disconnected
* \param context     Object representing librealsense session
* \param[in] callback function pointer to register as per-notifications callback
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
*/
void rs2_set_devices_changed_callback(const rs2_context* context, rs2_devices_changed_callback_ptr callback, void* user, rs2_error** error);

/**
 * Create a new device and add it to the context
 * \param ctx   The context to which the new device will be added
 * \param file  The file from which the device should be created
 * \param[out] error     If non-null, receives any error that occurs during this call, otherwise, errors are ignored
 * @return  A pointer to a device that plays data from the file, or null in case of failure
 */
rs2_device* rs2_context_add_device(rs2_context* ctx, const char* file, rs2_error** error);
    
/**
 * Add an instance of software device to the context
 * \param ctx   The context to which the new device will be added
 * \param dev   Instance of software device to register into the context
 * \param[out] error     If non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
void rs2_context_add_software_device(rs2_context* ctx, rs2_device* dev, rs2_error** error);

/**
 * Removes a playback device from the context, if exists
 * \param[in]  ctx       The context from which the device should be removed
 * \param[in]  file      The file name that was used to add the device
 * \param[out] error     If non-null, receives any error that occurs during this call, otherwise, errors are ignored
 */
void rs2_context_remove_device(rs2_context* ctx, const char* file, rs2_error** error);

/**
 * Removes tracking module.
 * function query_devices() locks the tracking module in the tm_context object. 
 * If the tracking module device is not used it should be removed using this function, so that other applications could find it.
 * This function can be used both before the call to query_device() to prevent enabling tracking modules or afterwards to 
 * release them.
 */
void rs2_context_unload_tracking_module(rs2_context* ctx, rs2_error** error);

/**
* create a static snapshot of all connected devices at the time of the call
* \param context     Object representing librealsense session
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            the list of devices, should be released by rs2_delete_device_list
*/
rs2_device_list* rs2_query_devices(const rs2_context* context, rs2_error** error);

#define RS2_PRODUCT_LINE_ANY            0xff
#define RS2_PRODUCT_LINE_ANY_INTEL      0xfe
#define RS2_PRODUCT_LINE_NON_INTEL      0x01
#define RS2_PRODUCT_LINE_D400           0x02
#define RS2_PRODUCT_LINE_SR300          0x04
#define RS2_PRODUCT_LINE_L500           0x08
#define RS2_PRODUCT_LINE_T200           0x10
#define RS2_PRODUCT_LINE_DEPTH      (RS2_PRODUCT_LINE_L500 | RS2_PRODUCT_LINE_SR300 | RS2_PRODUCT_LINE_D400)
#define RS2_PRODUCT_LINE_TRACKING   RS2_PRODUCT_LINE_T200

/**
* create a static snapshot of all connected devices at the time of the call
* \param context     Object representing librealsense session
* \param product_mask Controls what kind of devices will be returned
* \param[out] error  if non-null, receives any error that occurs during this call, otherwise, errors are ignored
* \return            the list of devices, should be released by rs2_delete_device_list
*/
rs2_device_list* rs2_query_devices_ex(const rs2_context* context, int product_mask, rs2_error** error);

/**
* \brief Creates RealSense device_hub .
* \param[in] context The context for the device hub
* \param[out] error  If non-null, receives any error that occurs during this call, otherwise, errors are ignored.
* \return            Device hub object
*/
rs2_device_hub* rs2_create_device_hub(const rs2_context* context, rs2_error** error);

/**
* \brief Frees the relevant device hub object.
* \param[in] hub Object that is no longer needed
*/
void rs2_delete_device_hub(const rs2_device_hub* hub);

/**
* If any device is connected return it, otherwise wait until next RealSense device connects.
* Calling this method multiple times will cycle through connected devices
* \param[in] ctx The context to creat the device
* \param[in] hub The device hub object
* \param[out] error  If non-null, receives any error that occurs during this call, otherwise, errors are ignored.
* \return            device object
*/
rs2_device* rs2_device_hub_wait_for_device(const rs2_device_hub* hub, rs2_error** error);

/**
* Checks if device is still connected
* \param[in] hub The device hub object
* \param[in] device The device
* \param[out] error  If non-null, receives any error that occurs during this call, otherwise, errors are ignored.
* \return            1 if the device is connected, 0 otherwise
*/
int rs2_device_hub_is_device_connected(const rs2_device_hub* hub, const rs2_device* device, rs2_error** error);


#ifdef __cplusplus
}
#endif
#endif
