/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (C) 2010-2012 Ken Tossell
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the author nor other contributors may be
*     used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/
/**
 * @defgroup device Device handling and enumeration
 * @brief Support for finding, inspecting and opening UVC devices
 */

#include "libuvc.h"
#include "libuvc_internal.h"

int uvc_already_open(uvc_context_t *ctx, struct libusb_device *usb_dev);
void uvc_free_devh(uvc_device_handle_t *devh);

uvc_error_t uvc_get_device_info(uvc_device_t *dev, uvc_device_info_t **info);
uvc_error_t uvc_get_device_info2(uvc_device_t *dev, uvc_device_info_t **info, int camera_number);
void uvc_free_device_info(uvc_device_info_t *info);

uvc_error_t uvc_scan_control(uvc_device_t *dev, uvc_device_info_t *info);
uvc_error_t uvc_parse_vc(uvc_device_t *dev,
             uvc_device_info_t *info,
             const unsigned char *block, size_t block_size);
uvc_error_t uvc_parse_vc_extension_unit(uvc_device_t *dev,
                    uvc_device_info_t *info,
                    const unsigned char *block,
                    size_t block_size);
uvc_error_t uvc_parse_vc_header(uvc_device_t *dev,
                uvc_device_info_t *info,
                const unsigned char *block, size_t block_size);
uvc_error_t uvc_parse_vc_input_terminal(uvc_device_t *dev,
                    uvc_device_info_t *info,
                    const unsigned char *block,
                    size_t block_size);
uvc_error_t uvc_parse_vc_processing_unit(uvc_device_t *dev,
                     uvc_device_info_t *info,
                     const unsigned char *block,
                     size_t block_size);

uvc_error_t uvc_scan_streaming(uvc_device_t *dev,
                   uvc_device_info_t *info,
                   int interface_idx);
uvc_error_t uvc_parse_vs(uvc_device_t *dev,
             uvc_device_info_t *info,
             uvc_streaming_interface_t *stream_if,
             const unsigned char *block, size_t block_size);
uvc_error_t uvc_parse_vs_format_uncompressed(uvc_streaming_interface_t *stream_if,
                         const unsigned char *block,
                         size_t block_size);
uvc_error_t uvc_parse_vs_format_mjpeg(uvc_streaming_interface_t *stream_if,
                         const unsigned char *block,
                         size_t block_size);
uvc_error_t uvc_parse_vs_frame_uncompressed(uvc_streaming_interface_t *stream_if,
                        const unsigned char *block,
                        size_t block_size);
uvc_error_t uvc_parse_vs_frame_format(uvc_streaming_interface_t *stream_if,
                        const unsigned char *block,
                        size_t block_size);
uvc_error_t uvc_parse_vs_frame_frame(uvc_streaming_interface_t *stream_if,
                        const unsigned char *block,
                        size_t block_size);
uvc_error_t uvc_parse_vs_input_header(uvc_streaming_interface_t *stream_if,
                      const unsigned char *block,
                      size_t block_size);

void LIBUSB_CALL _uvc_status_callback(struct libusb_transfer *transfer);

/** @internal
 * @brief Test whether the specified USB device has been opened as a UVC device
 * @ingroup device
 *
 * @param ctx Context in which to search for the UVC device
 * @param usb_dev USB device to find
 * @return true if the device is open in this context
 */
int uvc_already_open(uvc_context_t *ctx, struct libusb_device *usb_dev) {
  uvc_device_handle_t *devh;

  DL_FOREACH(ctx->open_devices, devh) {
    if (usb_dev == devh->dev->usb_dev)
      return 1;
  }

  return 0;
}

/** @brief Finds a camera identified by vendor, product and/or serial number
 * @ingroup device
 *
 * @param[in] ctx UVC context in which to search for the camera
 * @param[out] dev Reference to the camera, or NULL if not found
 * @param[in] vid Vendor ID number, optional
 * @param[in] pid Product ID number, optional
 * @param[in] sn Serial number or NULL
 * @return Error finding device or UVC_SUCCESS
 */
uvc_error_t uvc_find_device(
    uvc_context_t *ctx, uvc_device_t **dev,
    int vid, int pid, const char *sn) {
  uvc_error_t ret = UVC_SUCCESS;

  uvc_device_t **list;
  uvc_device_t *test_dev;
  int dev_idx;
  int found_dev;

  UVC_ENTER();

  ret = uvc_get_device_list(ctx, &list);

  if (ret != UVC_SUCCESS) {
    UVC_EXIT(ret);
    return ret;
  }

  dev_idx = 0;
  found_dev = 0;

  while (!found_dev && (test_dev = list[dev_idx++]) != NULL) {
    uvc_device_descriptor_t *desc;

    if (uvc_get_device_descriptor(test_dev, &desc) != UVC_SUCCESS)
      continue;

    if ((!vid || desc->idVendor == vid)
        && (!pid || desc->idProduct == pid)
        && (!sn || (desc->serialNumber && !strcmp(desc->serialNumber, sn))))
      found_dev = 1;

    uvc_free_device_descriptor(desc);
  }

  if (found_dev)
    uvc_ref_device(test_dev);

  uvc_free_device_list(list, 1);

  if (found_dev) {
    *dev = test_dev;
    UVC_EXIT(UVC_SUCCESS);
    return UVC_SUCCESS;
  } else {
    UVC_EXIT(UVC_ERROR_NO_DEVICE);
    return UVC_ERROR_NO_DEVICE;
  }
}

/** @brief Get the number of the bus to which the device is attached
 * @ingroup device
 */
uint8_t uvc_get_bus_number(uvc_device_t *dev) {
  return libusb_get_bus_number(dev->usb_dev);
}

/** @brief Get the number assigned to the device within its bus
 * @ingroup device
 */
uint8_t uvc_get_device_address(uvc_device_t *dev) {
  return libusb_get_device_address(dev->usb_dev);
}


/** @brief Open a UVC device, defaulting to the first interface found
 * @ingroup device
 *
 * @param dev Device to open
 * @param[out] devh Handle on opened device
 * @return Error opening device or SUCCESS
 */

uvc_error_t uvc_open(
    uvc_device_t *dev,
    uvc_device_handle_t **devh) {
  return uvc_open2(dev, devh, 0);
}

/** @brief Open a UVC device, specifying the camera number, zero-based
 * @ingroup device
 *
 * @param dev Device to open
 * @param [out] devh Handle on opened device
 * @param camera_number Camera to open
 * @return Error opening device or SUCCESS
 */

uvc_error_t uvc_open2(
    uvc_device_t *dev,
    uvc_device_handle_t **devh,
    int camera_number) {
  uvc_error_t ret;
  struct libusb_device_handle *usb_devh;
  uvc_device_handle_t *internal_devh;
  struct libusb_device_descriptor desc;

  UVC_ENTER();

  ret = libusb_open(dev->usb_dev, &usb_devh);
  UVC_DEBUG("libusb_open() = %d", ret);

  if (ret != UVC_SUCCESS) {
    UVC_EXIT(ret);
    return ret;
  }

  uvc_ref_device(dev);

  internal_devh = calloc(1, sizeof(*internal_devh));
  internal_devh->dev = dev;
  internal_devh->usb_devh = usb_devh;

  ret = uvc_get_device_info2(dev, &(internal_devh->info), camera_number);

  if (ret != UVC_SUCCESS)
    goto fail;

  UVC_DEBUG("claiming control interface %d", internal_devh->info->ctrl_if.bInterfaceNumber);
  ret = uvc_claim_if(internal_devh, internal_devh->info->ctrl_if.bInterfaceNumber);
  if (ret != UVC_SUCCESS)
    goto fail;

  libusb_get_device_descriptor(dev->usb_dev, &desc);
  internal_devh->is_isight = (desc.idVendor == 0x05ac && desc.idProduct == 0x8501);

  if (internal_devh->info->ctrl_if.bEndpointAddress) {
    internal_devh->status_xfer = libusb_alloc_transfer(0);
    if (!internal_devh->status_xfer) {
      ret = UVC_ERROR_NO_MEM;
      goto fail;
    }

    libusb_fill_interrupt_transfer(internal_devh->status_xfer,
                                   usb_devh,
                                   internal_devh->info->ctrl_if.bEndpointAddress,
                                   internal_devh->status_buf,
                                   sizeof(internal_devh->status_buf),
                                   _uvc_status_callback,
                                   internal_devh,
                                   0);
    ret = libusb_submit_transfer(internal_devh->status_xfer);
    UVC_DEBUG("libusb_submit_transfer() = %d", ret);

    if (ret) {
      UVC_DEBUG("device has a status interrupt endpoint, but unable to read from it");
      goto fail;
    }
  }

  if (dev->ctx->own_usb_ctx && dev->ctx->open_devices == NULL) {
    /* Since this is our first device, we need to spawn the event handler thread */
    uvc_start_handler_thread(dev->ctx);
  }

  DL_APPEND(dev->ctx->open_devices, internal_devh);
  *devh = internal_devh;

  UVC_EXIT(ret);

  return ret;

 fail:
  if ( internal_devh->info ) {
    uvc_release_if(internal_devh, internal_devh->info->ctrl_if.bInterfaceNumber);
  }
  libusb_close(usb_devh);
  uvc_unref_device(dev);
  uvc_free_devh(internal_devh);

  UVC_EXIT(ret);

  return ret;
}

/**
 * @internal
 * @brief Parses the complete device descriptor for a device. Defaults to using the first camera interface found on a device.
 * @ingroup device
 * @note Free *info with uvc_free_device_info when you're done
 *
 * @param dev Device to parse descriptor for
 * @param info Where to store a pointer to the new info struct
 * 
 */
uvc_error_t uvc_get_device_info(uvc_device_t *dev,
                uvc_device_info_t **info) {
  return uvc_get_device_info2(dev, info, 0);
}

/**
 * @internal
 * @brief Parses the complete device descriptor for a device
 * @ingroup device
 * @note Free *info with uvc_free_device_info when you're done
 *
 * @param dev Device to parse descriptor for
 * @param info Where to store a pointer to the new info struct
 * @param camera_number Index of the camera to open, 0-based
 */

uvc_error_t uvc_get_device_info2(uvc_device_t *dev,
                uvc_device_info_t **info,
                int camera_number) {
  uvc_error_t ret;
  uvc_device_info_t *internal_info;

  UVC_ENTER();

  internal_info = calloc(1, sizeof(*internal_info));
  if (!internal_info) {
    UVC_EXIT(UVC_ERROR_NO_MEM);
    return UVC_ERROR_NO_MEM;
  }

  if (libusb_get_config_descriptor(dev->usb_dev,
                   0,
                   &(internal_info->config)) != 0) {
    free(internal_info);
    UVC_EXIT(UVC_ERROR_IO);
    return UVC_ERROR_IO;
  }

  if (camera_number*2 > internal_info->config->bNumInterfaces)
    {
      free(internal_info);
      UVC_EXIT(UVC_ERROR_NO_DEVICE);
      return UVC_ERROR_NO_DEVICE;
    }

  //Which camera interface to pick up
  internal_info->camera_number = camera_number;


  ret = uvc_scan_control(dev, internal_info);
  if (ret != UVC_SUCCESS) {
    uvc_free_device_info(internal_info);
    UVC_EXIT(ret);
    return ret;
  }

  *info = internal_info;

  UVC_EXIT(ret);
  return ret;
}

/**
 * @internal
 * @brief Frees the device descriptor for a device
 * @ingroup device
 *
 * @param info Which device info block to free
 */
void uvc_free_device_info(uvc_device_info_t *info) {
  uvc_input_terminal_t *input_term, *input_term_tmp;
  uvc_processing_unit_t *proc_unit, *proc_unit_tmp;
  uvc_extension_unit_t *ext_unit, *ext_unit_tmp;

  uvc_streaming_interface_t *stream_if, *stream_if_tmp;
  uvc_format_desc_t *format, *format_tmp;
  uvc_frame_desc_t *frame, *frame_tmp;

  UVC_ENTER();

  DL_FOREACH_SAFE(info->ctrl_if.input_term_descs, input_term, input_term_tmp) {
    DL_DELETE(info->ctrl_if.input_term_descs, input_term);
    free(input_term);
  }

  DL_FOREACH_SAFE(info->ctrl_if.processing_unit_descs, proc_unit, proc_unit_tmp) {
    DL_DELETE(info->ctrl_if.processing_unit_descs, proc_unit);
    free(proc_unit);
  }

  DL_FOREACH_SAFE(info->ctrl_if.extension_unit_descs, ext_unit, ext_unit_tmp) {
    DL_DELETE(info->ctrl_if.extension_unit_descs, ext_unit);
    free(ext_unit);
  }

  DL_FOREACH_SAFE(info->stream_ifs, stream_if, stream_if_tmp) {
    DL_FOREACH_SAFE(stream_if->format_descs, format, format_tmp) {
      DL_FOREACH_SAFE(format->frame_descs, frame, frame_tmp) {
        if (frame->intervals)
          free(frame->intervals);

        DL_DELETE(format->frame_descs, frame);
        free(frame);
      }

      DL_DELETE(stream_if->format_descs, format);
      free(format);
    }

    DL_DELETE(info->stream_ifs, stream_if);
    free(stream_if);
  }

  if (info->config)
    libusb_free_config_descriptor(info->config);

  free(info);

  UVC_EXIT_VOID();
}

/**
 * @brief Get a descriptor that contains the general information about
 * a device
 * @ingroup device
 *
 * Free *desc with uvc_free_device_descriptor when you're done.
 *
 * @param dev Device to fetch information about
 * @param[out] desc Descriptor structure
 * @return Error if unable to fetch information, else SUCCESS
 */
uvc_error_t uvc_get_device_descriptor(
    uvc_device_t *dev,
    uvc_device_descriptor_t **desc) {
  uvc_device_descriptor_t *desc_internal;
  struct libusb_device_descriptor usb_desc;
  struct libusb_device_handle *usb_devh;
  uvc_error_t ret;

  UVC_ENTER();

  ret = libusb_get_device_descriptor(dev->usb_dev, &usb_desc);

  if (ret != UVC_SUCCESS) {
    UVC_EXIT(ret);
    return ret;
  }

  desc_internal = calloc(1, sizeof(*desc_internal));
  desc_internal->idVendor = usb_desc.idVendor;
  desc_internal->idProduct = usb_desc.idProduct;

  if (libusb_open(dev->usb_dev, &usb_devh) == 0) {
    unsigned char buf[64];

    int bytes = libusb_get_string_descriptor_ascii(
        usb_devh, usb_desc.iSerialNumber, buf, sizeof(buf));

    if (bytes > 0)
      desc_internal->serialNumber = strdup((const char*) buf);

    bytes = libusb_get_string_descriptor_ascii(
        usb_devh, usb_desc.iManufacturer, buf, sizeof(buf));

    if (bytes > 0)
      desc_internal->manufacturer = strdup((const char*) buf);

    bytes = libusb_get_string_descriptor_ascii(
        usb_devh, usb_desc.iProduct, buf, sizeof(buf));

    if (bytes > 0)
      desc_internal->product = strdup((const char*) buf);

    libusb_close(usb_devh);
  } else {
    UVC_DEBUG("can't open device %04x:%04x, not fetching serial etc.",
          usb_desc.idVendor, usb_desc.idProduct);
  }

  *desc = desc_internal;

  UVC_EXIT(ret);
  return ret;
}

/**
 * @brief Frees a device descriptor created with uvc_get_device_descriptor
 * @ingroup device
 *
 * @param desc Descriptor to free
 */
void uvc_free_device_descriptor(
    uvc_device_descriptor_t *desc) {
  UVC_ENTER();

  if (desc->serialNumber)
    free((void*) desc->serialNumber);

  if (desc->manufacturer)
    free((void*) desc->manufacturer);

  if (desc->product)
    free((void*) desc->product);

  free(desc);

  UVC_EXIT_VOID();
}

/**
 * @brief Get a list of the UVC devices attached to the system
 * @ingroup device
 *
 * @note Free the list with uvc_free_device_list when you're done.
 *
 * @param ctx UVC context in which to list devices
 * @param list List of uvc_device structures
 * @return Error if unable to list devices, else SUCCESS
 */
uvc_error_t uvc_get_device_list(
    uvc_context_t *ctx,
    uvc_device_t ***list) {
  uvc_error_t ret;
  struct libusb_device **usb_dev_list;
  struct libusb_device *usb_dev;
  int num_usb_devices;

  uvc_device_t **list_internal;
  int num_uvc_devices;

  /* per device */
  int dev_idx;
  struct libusb_device_handle *usb_devh;
  struct libusb_config_descriptor *config;
  struct libusb_device_descriptor desc;
  uint8_t got_interface;

  /* per interface */
  int interface_idx;
  const struct libusb_interface *interface;

  /* per altsetting */
  int altsetting_idx;
  const struct libusb_interface_descriptor *if_desc;

  UVC_ENTER();

  num_usb_devices = libusb_get_device_list(ctx->usb_ctx, &usb_dev_list);

  if (num_usb_devices < 0) {
    UVC_EXIT(UVC_ERROR_IO);
    return UVC_ERROR_IO;
  }

  list_internal = malloc(sizeof(*list_internal));
  *list_internal = NULL;

  num_uvc_devices = 0;
  dev_idx = -1;

  while ((usb_dev = usb_dev_list[++dev_idx]) != NULL) {
    usb_devh = NULL;
    got_interface = 0;

    if (libusb_get_config_descriptor(usb_dev, 0, &config) != 0)
      continue;

    if ( libusb_get_device_descriptor ( usb_dev, &desc ) != LIBUSB_SUCCESS )
      continue;

    // Special case for Imaging Source cameras
    if ( 0x199e == desc.idVendor && 0x8101 == desc.idProduct ) {
      got_interface = 1;
    } else {

      for (interface_idx = 0;
       !got_interface && interface_idx < config->bNumInterfaces;
       ++interface_idx) {
        interface = &config->interface[interface_idx];

        for (altsetting_idx = 0;
         !got_interface && altsetting_idx < interface->num_altsetting;
         ++altsetting_idx) {
      if_desc = &interface->altsetting[altsetting_idx];

      /* Video, Streaming */
      if (if_desc->bInterfaceClass == 14 && if_desc->bInterfaceSubClass == 2) {
        got_interface = 1;
      }
        }
      }
    }

    libusb_free_config_descriptor(config);

    if (got_interface) {
      uvc_device_t *uvc_dev = malloc(sizeof(*uvc_dev));
      uvc_dev->ctx = ctx;
      uvc_dev->ref = 0;
      uvc_dev->usb_dev = usb_dev;
      uvc_ref_device(uvc_dev);

      num_uvc_devices++;
      list_internal = realloc(list_internal, (num_uvc_devices + 1) * sizeof(*list_internal));

      list_internal[num_uvc_devices - 1] = uvc_dev;
      list_internal[num_uvc_devices] = NULL;

      UVC_DEBUG("    UVC: %d", dev_idx);
    } else {
      UVC_DEBUG("non-UVC: %d", dev_idx);
    }
  }

  libusb_free_device_list(usb_dev_list, 1);

  *list = list_internal;

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/**
 * @brief Frees a list of device structures created with uvc_get_device_list.
 * @ingroup device
 *
 * @param list Device list to free
 * @param unref_devices Decrement the reference counter for each device
 * in the list, and destroy any entries that end up with zero references
 */
void uvc_free_device_list(uvc_device_t **list, uint8_t unref_devices) {
  uvc_device_t *dev;
  int dev_idx = 0;

  UVC_ENTER();

  if (unref_devices) {
    while ((dev = list[dev_idx++]) != NULL) {
      uvc_unref_device(dev);
    }
  }

  free(list);

  UVC_EXIT_VOID();
}

/**
 * @brief Get the uvc_device_t corresponding to an open device
 * @ingroup device
 *
 * @note Unref the uvc_device_t when you're done with it
 *
 * @param devh Device handle to an open UVC device
 */
uvc_device_t *uvc_get_device(uvc_device_handle_t *devh) {
  uvc_ref_device(devh->dev);
  return devh->dev;
}

/**
 * @brief Get the underlying libusb device handle for an open device
 * @ingroup device
 *
 * This can be used to access other interfaces on the same device, e.g.
 * a webcam microphone.
 *
 * @note The libusb device handle is only valid while the UVC device is open;
 * it will be invalidated upon calling uvc_close.
 *
 * @param devh UVC device handle to an open device
 */
libusb_device_handle *uvc_get_libusb_handle(uvc_device_handle_t *devh) {
  return devh->usb_devh;
}

/**
 * @brief Get input terminal descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_input_terminal_t *uvc_get_input_terminals(uvc_device_handle_t *devh) {
  return devh->info->ctrl_if.input_term_descs;
}

/**
 * @brief Get output terminal descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_output_terminal_t *uvc_get_output_terminals(uvc_device_handle_t *devh) {
  return NULL; /* @todo */
}

/**
 * @brief Get processing unit descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_processing_unit_t *uvc_get_processing_units(uvc_device_handle_t *devh) {
  return devh->info->ctrl_if.processing_unit_descs;
}

/**
 * @brief Get extension unit descriptors for the open device.
 *
 * @note Do not modify the returned structure.
 * @note The returned structure is part of a linked list. Iterate through
 *       it by using the 'next' pointers.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_extension_unit_t *uvc_get_extension_units(uvc_device_handle_t *devh) {
  return devh->info->ctrl_if.extension_unit_descs;
}

/**
 * @brief Increment the reference count for a device
 * @ingroup device
 *
 * @param dev Device to reference
 */
void uvc_ref_device(uvc_device_t *dev) {
  UVC_ENTER();

  dev->ref++;
  libusb_ref_device(dev->usb_dev);

  UVC_EXIT_VOID();
}

/**
 * @brief Decrement the reference count for a device
 * @ingropu device
 * @note If the count reaches zero, the device will be discarded
 *
 * @param dev Device to unreference
 */
void uvc_unref_device(uvc_device_t *dev) {
  UVC_ENTER();

  libusb_unref_device(dev->usb_dev);
  dev->ref--;

  if (dev->ref == 0)
    free(dev);

  UVC_EXIT_VOID();
}

/** @internal
 * Claim a UVC interface, detaching the kernel driver if necessary.
 * @ingroup device
 *
 * @param devh UVC device handle
 * @param idx UVC interface index
 */
uvc_error_t uvc_claim_if(uvc_device_handle_t *devh, int idx) {
  int ret;

  UVC_ENTER();

  /* Tell libusb to detach any active kernel drivers. libusb will keep track of whether
   * it found a kernel driver for this interface. */
  ret = libusb_detach_kernel_driver(devh->usb_devh, idx);

  if (ret == UVC_SUCCESS || ret == LIBUSB_ERROR_NOT_FOUND || ret == LIBUSB_ERROR_NOT_SUPPORTED) {
    UVC_DEBUG("claiming interface %d", idx);
    ret = libusb_claim_interface(devh->usb_devh, idx);
  } else {
    UVC_DEBUG("not claiming interface %d: unable to detach kernel driver (%s)",
              idx, uvc_strerror(ret));
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * Release a UVC interface.
 * @ingroup device
 *
 * @param devh UVC device handle
 * @param idx UVC interface index
 */
uvc_error_t uvc_release_if(uvc_device_handle_t *devh, int idx) {
  int ret;

  UVC_ENTER();
  UVC_DEBUG("releasing interface %d", idx);
  /* libusb_release_interface *should* reset the alternate setting to the first available,
     but sometimes (e.g. on Darwin) it doesn't. Thus, we do it explicitly here.
     This is needed to de-initialize certain cameras. */
  libusb_set_interface_alt_setting(devh->usb_devh, idx, 0);
  ret = libusb_release_interface(devh->usb_devh, idx);

  if (UVC_SUCCESS == ret) {
    /* Reattach any kernel drivers that were disabled when we claimed this interface */
    ret = libusb_attach_kernel_driver(devh->usb_devh, idx);

    if (ret == UVC_SUCCESS) {
      UVC_DEBUG("reattached kernel driver to interface %d", idx);
    } else if (ret == LIBUSB_ERROR_NOT_FOUND || ret == LIBUSB_ERROR_NOT_SUPPORTED) {
      ret = UVC_SUCCESS;  /* NOT_FOUND and NOT_SUPPORTED are OK: nothing to do */
    } else {
      UVC_DEBUG("error reattaching kernel driver to interface %d: %s",
                idx, uvc_strerror(ret));
    }
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * Find a device's VideoControl interface and process its descriptor
 * @ingroup device
 */
uvc_error_t uvc_scan_control(uvc_device_t *dev, uvc_device_info_t *info) {
  const struct libusb_interface_descriptor *if_desc;
  uvc_error_t parse_ret, ret;
  int interface_idx;
  const unsigned char *buffer;
  size_t buffer_left, block_size;

  UVC_ENTER();

  ret = UVC_SUCCESS;
  if_desc = NULL;

  //Start looking for the control interface at the given camera number

  for (interface_idx = info->camera_number*2; interface_idx < info->config->bNumInterfaces; ++interface_idx) {
    if_desc = &info->config->interface[interface_idx].altsetting[0];

    if (if_desc->bInterfaceClass == 14 && if_desc->bInterfaceSubClass == 1) // Video, Control
      break;

    // Another TIS camera hack.
    if ( if_desc->bInterfaceClass == 255 && if_desc->bInterfaceSubClass == 1 ) {
      uvc_device_descriptor_t* dev_desc;
      int haveTISCamera = 0;
      uvc_get_device_descriptor ( dev, &dev_desc );
      if ( dev_desc->idVendor == 0x199e && dev_desc->idProduct == 0x8101 ) {
        haveTISCamera = 1;
      }
      uvc_free_device_descriptor ( dev_desc );
      if ( haveTISCamera ) {
        break;
      }
    }

    if_desc = NULL;
  }

  if (if_desc == NULL) {
    UVC_EXIT(UVC_ERROR_INVALID_DEVICE);
    return UVC_ERROR_INVALID_DEVICE;
  }

  info->ctrl_if.bInterfaceNumber = interface_idx;
  if (if_desc->bNumEndpoints != 0) {
    info->ctrl_if.bEndpointAddress = if_desc->endpoint[0].bEndpointAddress;
  }

  buffer = if_desc->extra;
  buffer_left = if_desc->extra_length;

  while (buffer_left >= 3) { // parseX needs to see buf[0,2] = length,type
    block_size = buffer[0];
    parse_ret = uvc_parse_vc(dev, info, buffer, block_size);

    if (parse_ret != UVC_SUCCESS) {
      ret = parse_ret;
      break;
    }

    buffer_left -= block_size;
    buffer += block_size;
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * @brief Parse a VideoControl header.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_header(uvc_device_t *dev,
                uvc_device_info_t *info,
                const unsigned char *block, size_t block_size) {
  size_t i;
  uvc_error_t scan_ret, ret = UVC_SUCCESS;

  UVC_ENTER();

  /*
  int uvc_version;
  uvc_version = (block[4] >> 4) * 1000 + (block[4] & 0x0f) * 100
    + (block[3] >> 4) * 10 + (block[3] & 0x0f);
  */

  info->ctrl_if.bcdUVC = SW_TO_SHORT(&block[3]);

  switch (info->ctrl_if.bcdUVC) {
  case 0x0100:
  case 0x010a:
  case 0x0110:
    break;
  default:
    UVC_EXIT(UVC_ERROR_NOT_SUPPORTED);
    return UVC_ERROR_NOT_SUPPORTED;
  }

  for (i = 12; i < block_size; ++i) {
    scan_ret = uvc_scan_streaming(dev, info, block[i]);
    if (scan_ret != UVC_SUCCESS) {
      ret = scan_ret;
      break;
    }
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * @brief Parse a VideoControl input terminal.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_input_terminal(uvc_device_t *dev,
                    uvc_device_info_t *info,
                    const unsigned char *block, size_t block_size) {
  uvc_input_terminal_t *term;
  size_t i;

  UVC_ENTER();

  /* only supporting camera-type input terminals */
  if (SW_TO_SHORT(&block[4]) != UVC_ITT_CAMERA) {
    UVC_EXIT(UVC_SUCCESS);
    return UVC_SUCCESS;
  }

  term = calloc(1, sizeof(*term));

  term->bTerminalID = block[3];
  term->wTerminalType = SW_TO_SHORT(&block[4]);
  term->wObjectiveFocalLengthMin = SW_TO_SHORT(&block[8]);
  term->wObjectiveFocalLengthMax = SW_TO_SHORT(&block[10]);
  term->wOcularFocalLength = SW_TO_SHORT(&block[12]);

  for (i = 14 + block[14]; i >= 15; --i)
    term->bmControls = block[i] + (term->bmControls << 8);

  DL_APPEND(info->ctrl_if.input_term_descs, term);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoControl processing unit.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_processing_unit(uvc_device_t *dev,
                     uvc_device_info_t *info,
                     const unsigned char *block, size_t block_size) {
  uvc_processing_unit_t *unit;
  size_t i;

  UVC_ENTER();

  unit = calloc(1, sizeof(*unit));
  unit->bUnitID = block[3];
  unit->bSourceID = block[4];
  
  for (i = 7 + block[7]; i >= 8; --i)
    unit->bmControls = block[i] + (unit->bmControls << 8);
  
  DL_APPEND(info->ctrl_if.processing_unit_descs, unit);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoControl extension unit.
 * @ingroup device
 */
uvc_error_t uvc_parse_vc_extension_unit(uvc_device_t *dev,
                    uvc_device_info_t *info,
                    const unsigned char *block, size_t block_size) {
  uvc_extension_unit_t *unit = calloc(1, sizeof(*unit));
  const uint8_t *start_of_controls;
  int size_of_controls, num_in_pins;
  int i;

  UVC_ENTER();

  unit->bUnitID = block[3];
  memcpy(unit->guidExtensionCode, &block[4], 16);
  
  num_in_pins = block[21];
  size_of_controls = block[22 + num_in_pins];
  start_of_controls = &block[23 + num_in_pins];
  
  for (i = size_of_controls - 1; i >= 0; --i)
    unit->bmControls = start_of_controls[i] + (unit->bmControls << 8);
  
  DL_APPEND(info->ctrl_if.extension_unit_descs, unit);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * Process a single VideoControl descriptor block
 * @ingroup device
 */
uvc_error_t uvc_parse_vc(
    uvc_device_t *dev,
    uvc_device_info_t *info,
    const unsigned char *block, size_t block_size) {
  int descriptor_subtype;
  uvc_error_t ret = UVC_SUCCESS;

  UVC_ENTER();

  if (block[1] != 36) { // not a CS_INTERFACE descriptor??
    UVC_EXIT(UVC_SUCCESS);
    return UVC_SUCCESS; // UVC_ERROR_INVALID_DEVICE;
  }

  descriptor_subtype = block[2];

  switch (descriptor_subtype) {
  case UVC_VC_HEADER:
    ret = uvc_parse_vc_header(dev, info, block, block_size);
    break;
  case UVC_VC_INPUT_TERMINAL:
    ret = uvc_parse_vc_input_terminal(dev, info, block, block_size);
    break;
  case UVC_VC_OUTPUT_TERMINAL:
    break;
  case UVC_VC_SELECTOR_UNIT:
    break;
  case UVC_VC_PROCESSING_UNIT:
    ret = uvc_parse_vc_processing_unit(dev, info, block, block_size);
    break;
  case UVC_VC_EXTENSION_UNIT:
    ret = uvc_parse_vc_extension_unit(dev, info, block, block_size);
    break;
  default:
    ret = UVC_ERROR_INVALID_DEVICE;
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * Process a VideoStreaming interface
 * @ingroup device
 */
uvc_error_t uvc_scan_streaming(uvc_device_t *dev,
                   uvc_device_info_t *info,
                   int interface_idx) {
  const struct libusb_interface_descriptor *if_desc;
  const unsigned char *buffer;
  size_t buffer_left, block_size;
  uvc_error_t ret, parse_ret;
  uvc_streaming_interface_t *stream_if;

  UVC_ENTER();

  ret = UVC_SUCCESS;

  if_desc = &(info->config->interface[interface_idx].altsetting[0]);
  buffer = if_desc->extra;
  buffer_left = if_desc->extra_length;

  stream_if = calloc(1, sizeof(*stream_if));
  stream_if->parent = info;
  stream_if->bInterfaceNumber = if_desc->bInterfaceNumber;
  DL_APPEND(info->stream_ifs, stream_if);

  while (buffer_left >= 3) {
    block_size = buffer[0];
    parse_ret = uvc_parse_vs(dev, info, stream_if, buffer, block_size);

    if (parse_ret != UVC_SUCCESS) {
      ret = parse_ret;
      break;
    }

    buffer_left -= block_size;
    buffer += block_size;
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * @brief Parse a VideoStreaming header block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_input_header(uvc_streaming_interface_t *stream_if,
                      const unsigned char *block,
                      size_t block_size) {
  UVC_ENTER();

  stream_if->bEndpointAddress = block[6] & 0x8f;
  stream_if->bTerminalLink = block[8];

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming uncompressed format block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_format_uncompressed(uvc_streaming_interface_t *stream_if,
                         const unsigned char *block,
                         size_t block_size) {
  UVC_ENTER();

  uvc_format_desc_t *format = calloc(1, sizeof(*format));

  format->parent = stream_if;
  format->bDescriptorSubtype = block[2];
  format->bFormatIndex = block[3];
  //format->bmCapabilities = block[4];
  //format->bmFlags = block[5];
  memcpy(format->guidFormat, &block[5], 16);
  format->bBitsPerPixel = block[21];
  format->bDefaultFrameIndex = block[22];
  format->bAspectRatioX = block[23];
  format->bAspectRatioY = block[24];
  format->bmInterlaceFlags = block[25];
  format->bCopyProtect = block[26];

  DL_APPEND(stream_if->format_descs, format);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming frame format block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_frame_format(uvc_streaming_interface_t *stream_if,
                         const unsigned char *block,
                         size_t block_size) {
  UVC_ENTER();

  uvc_format_desc_t *format = calloc(1, sizeof(*format));

  format->parent = stream_if;
  format->bDescriptorSubtype = block[2];
  format->bFormatIndex = block[3];
  format->bNumFrameDescriptors = block[4];
  memcpy(format->guidFormat, &block[5], 16);
  format->bBitsPerPixel = block[21];
  format->bDefaultFrameIndex = block[22];
  format->bAspectRatioX = block[23];
  format->bAspectRatioY = block[24];
  format->bmInterlaceFlags = block[25];
  format->bCopyProtect = block[26];
  format->bVariableSize = block[27];

  DL_APPEND(stream_if->format_descs, format);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming MJPEG format block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_format_mjpeg(uvc_streaming_interface_t *stream_if,
                         const unsigned char *block,
                         size_t block_size) {
  UVC_ENTER();

  uvc_format_desc_t *format = calloc(1, sizeof(*format));

  format->parent = stream_if;
  format->bDescriptorSubtype = block[2];
  format->bFormatIndex = block[3];
  memcpy(format->fourccFormat, "MJPG", 4);
  format->bmFlags = block[5];
  format->bBitsPerPixel = 0;
  format->bDefaultFrameIndex = block[6];
  format->bAspectRatioX = block[7];
  format->bAspectRatioY = block[8];
  format->bmInterlaceFlags = block[9];
  format->bCopyProtect = block[10];

  DL_APPEND(stream_if->format_descs, format);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming uncompressed frame block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_frame_frame(uvc_streaming_interface_t *stream_if,
                        const unsigned char *block,
                        size_t block_size) {
  uvc_format_desc_t *format;
  uvc_frame_desc_t *frame;

  const unsigned char *p;
  int i;

  UVC_ENTER();

  format = stream_if->format_descs->prev;
  frame = calloc(1, sizeof(*frame));

  frame->parent = format;

  frame->bDescriptorSubtype = block[2];
  frame->bFrameIndex = block[3];
  frame->bmCapabilities = block[4];
  frame->wWidth = block[5] + (block[6] << 8);
  frame->wHeight = block[7] + (block[8] << 8);
  frame->dwMinBitRate = DW_TO_INT(&block[9]);
  frame->dwMaxBitRate = DW_TO_INT(&block[13]);
  frame->dwDefaultFrameInterval = DW_TO_INT(&block[17]);
  frame->bFrameIntervalType = block[21];
  frame->dwBytesPerLine = DW_TO_INT(&block[22]);

  if (block[21] == 0) {
    frame->dwMinFrameInterval = DW_TO_INT(&block[26]);
    frame->dwMaxFrameInterval = DW_TO_INT(&block[30]);
    frame->dwFrameIntervalStep = DW_TO_INT(&block[34]);
  } else {
    frame->intervals = calloc(block[21] + 1, sizeof(frame->intervals[0]));
    p = &block[26];

    for (i = 0; i < block[21]; ++i) {
      frame->intervals[i] = DW_TO_INT(p);
      p += 4;
    }
    frame->intervals[block[21]] = 0;
  }

  DL_APPEND(format->frame_descs, frame);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * @brief Parse a VideoStreaming uncompressed frame block.
 * @ingroup device
 */
uvc_error_t uvc_parse_vs_frame_uncompressed(uvc_streaming_interface_t *stream_if,
                        const unsigned char *block,
                        size_t block_size) {
  uvc_format_desc_t *format;
  uvc_frame_desc_t *frame;

  const unsigned char *p;
  int i;

  UVC_ENTER();

  format = stream_if->format_descs->prev;
  frame = calloc(1, sizeof(*frame));

  frame->parent = format;

  frame->bDescriptorSubtype = block[2];
  frame->bFrameIndex = block[3];
  frame->bmCapabilities = block[4];
  frame->wWidth = block[5] + (block[6] << 8);
  frame->wHeight = block[7] + (block[8] << 8);
  frame->dwMinBitRate = DW_TO_INT(&block[9]);
  frame->dwMaxBitRate = DW_TO_INT(&block[13]);
  frame->dwMaxVideoFrameBufferSize = DW_TO_INT(&block[17]);
  frame->dwDefaultFrameInterval = DW_TO_INT(&block[21]);
  frame->bFrameIntervalType = block[25];

  if (block[25] == 0) {
    frame->dwMinFrameInterval = DW_TO_INT(&block[26]);
    frame->dwMaxFrameInterval = DW_TO_INT(&block[30]);
    frame->dwFrameIntervalStep = DW_TO_INT(&block[34]);
  } else {
    frame->intervals = calloc(block[25] + 1, sizeof(frame->intervals[0]));
    p = &block[26];

    for (i = 0; i < block[25]; ++i) {
      frame->intervals[i] = DW_TO_INT(p);
      p += 4;
    }
    frame->intervals[block[25]] = 0;
  }

  DL_APPEND(format->frame_descs, frame);

  UVC_EXIT(UVC_SUCCESS);
  return UVC_SUCCESS;
}

/** @internal
 * Process a single VideoStreaming descriptor block
 * @ingroup device
 */
uvc_error_t uvc_parse_vs(
    uvc_device_t *dev,
    uvc_device_info_t *info,
    uvc_streaming_interface_t *stream_if,
    const unsigned char *block, size_t block_size) {
  uvc_error_t ret;
  int descriptor_subtype;

  UVC_ENTER();

  ret = UVC_SUCCESS;
  descriptor_subtype = block[2];

  switch (descriptor_subtype) {
  case UVC_VS_INPUT_HEADER:
    ret = uvc_parse_vs_input_header(stream_if, block, block_size);
    break;
  case UVC_VS_FORMAT_UNCOMPRESSED:
    ret = uvc_parse_vs_format_uncompressed(stream_if, block, block_size);
    break;
  case UVC_VS_FORMAT_MJPEG:
    ret = uvc_parse_vs_format_mjpeg(stream_if, block, block_size);
    break;
  case UVC_VS_FRAME_UNCOMPRESSED:
  case UVC_VS_FRAME_MJPEG:
    ret = uvc_parse_vs_frame_uncompressed(stream_if, block, block_size);
    break;
  case UVC_VS_FORMAT_FRAME_BASED:
    ret = uvc_parse_vs_frame_format ( stream_if, block, block_size );
    break;
  case UVC_VS_FRAME_FRAME_BASED:
    ret = uvc_parse_vs_frame_frame ( stream_if, block, block_size );
    break;
  default:
    /** @todo handle JPEG and maybe still frames or even DV... */
    UVC_DEBUG ("unsupported descriptor subtype: %d\n", descriptor_subtype);
    break;
  }

  UVC_EXIT(ret);
  return ret;
}

/** @internal
 * @brief Free memory associated with a UVC device
 * @pre Streaming must be stopped, and threads must have died
 */
void uvc_free_devh(uvc_device_handle_t *devh) {
  UVC_ENTER();

  if (devh->info)
    uvc_free_device_info(devh->info);

  if (devh->status_xfer)
    libusb_free_transfer(devh->status_xfer);

  free(devh);

  UVC_EXIT_VOID();
}

/** @brief Close a device
 *
 * @ingroup device
 *
 * Ends any stream that's in progress.
 *
 * The device handle and frame structures will be invalidated.
 */
void uvc_close(uvc_device_handle_t *devh) {
  UVC_ENTER();
  uvc_context_t *ctx = devh->dev->ctx;

  if (devh->streams)
    uvc_stop_streaming(devh);

  uvc_release_if(devh, devh->info->ctrl_if.bInterfaceNumber);

  /* If we are managing the libusb context and this is the last open device,
   * then we need to cancel the handler thread. When we call libusb_close,
   * it'll cause a return from the thread's libusb_handle_events call, after
   * which the handler thread will check the flag we set and then exit. */
  if (ctx->own_usb_ctx && ctx->open_devices == devh && devh->next == NULL) {
    ctx->kill_handler_thread = 1;
    libusb_close(devh->usb_devh);
    pthread_join(ctx->handler_thread, NULL);
  } else {
    libusb_close(devh->usb_devh);
  }

  DL_DELETE(ctx->open_devices, devh);

  uvc_unref_device(devh->dev);

  uvc_free_devh(devh);

  UVC_EXIT_VOID();
}

/** @internal
 * @brief Get number of open devices
 */
size_t uvc_num_devices(uvc_context_t *ctx) {
  size_t count = 0;

  uvc_device_handle_t *devh;

  UVC_ENTER();

  DL_FOREACH(ctx->open_devices, devh) {
    count++;
  }

  UVC_EXIT((int) count);
  return count;
}

void uvc_process_status_xfer(uvc_device_handle_t *devh, struct libusb_transfer *transfer) {
  enum uvc_status_class status_class;
  uint8_t originator = 0, selector = 0, event = 0;
  enum uvc_status_attribute attribute = UVC_STATUS_ATTRIBUTE_UNKNOWN;
  void *data = NULL;
  size_t data_len = 0;

  UVC_ENTER();

  /* printf("Got transfer of aLen = %d\n", transfer->actual_length); */

  if (transfer->actual_length < 4) {
    UVC_DEBUG("Short read of status update (%d bytes)", transfer->actual_length);
    UVC_EXIT_VOID();
    return;
  }

  originator = transfer->buffer[1];

  switch (transfer->buffer[0] & 0x0f) {
  case 1: {  /* VideoControl interface */
    int found_entity = 0;
    struct uvc_input_terminal *input_terminal;
    struct uvc_processing_unit *processing_unit;

    if (transfer->actual_length < 5) {
      UVC_DEBUG("Short read of VideoControl status update (%d bytes)",
        transfer->actual_length);
      UVC_EXIT_VOID();
      return;
    }

    event = transfer->buffer[2];
    selector = transfer->buffer[3];

    if (originator == 0) {
      UVC_DEBUG("Unhandled update from VC interface");
      UVC_EXIT_VOID();
      return;  /* @todo VideoControl virtual entity interface updates */
    }

    if (event != 0) {
      UVC_DEBUG("Unhandled VC event %d", (int) event);
      UVC_EXIT_VOID();
      return;
    }

    /* printf("bSelector: %d\n", selector); */

    DL_FOREACH(devh->info->ctrl_if.input_term_descs, input_terminal) {
      if (input_terminal->bTerminalID == originator) {
        status_class = UVC_STATUS_CLASS_CONTROL_CAMERA;
        found_entity = 1;
        break;
      }
    }

    if (!found_entity) {
      DL_FOREACH(devh->info->ctrl_if.processing_unit_descs, processing_unit) {
        if (processing_unit->bUnitID == originator) {
          status_class = UVC_STATUS_CLASS_CONTROL_PROCESSING;
          found_entity = 1;
          break;
        }
      }
    }

    if (!found_entity) {
      UVC_DEBUG("Got status update for unknown VideoControl entity %d",
        (int) originator);
      UVC_EXIT_VOID();
      return;
    }

    attribute = transfer->buffer[4];
    data = transfer->buffer + 5;
    data_len = transfer->actual_length - 5;
    break;
  }
  case 2:  /* VideoStreaming interface */
    UVC_DEBUG("Unhandled update from VideoStreaming interface");
    UVC_EXIT_VOID();
    return;  /* @todo VideoStreaming updates */
  }

  UVC_DEBUG("Event: class=%d, event=%d, selector=%d, attribute=%d, data_len=%zd",
    status_class, event, selector, attribute, data_len);

  if(devh->status_cb) {
    UVC_DEBUG("Running user-supplied status callback");
    devh->status_cb(status_class,
                    event,
                    selector,
                    attribute,
                    data, data_len,
                    devh->status_user_ptr);
  }

  UVC_EXIT_VOID();
}

/** @internal
 * @brief Process asynchronous status updates from the device.
 */
void LIBUSB_CALL _uvc_status_callback(struct libusb_transfer *transfer) {
  UVC_ENTER();

  uvc_device_handle_t *devh = (uvc_device_handle_t *) transfer->user_data;

  switch (transfer->status) {
  case LIBUSB_TRANSFER_ERROR:
  case LIBUSB_TRANSFER_CANCELLED:
  case LIBUSB_TRANSFER_NO_DEVICE:
    UVC_DEBUG("retrying transfer, status = %s", libusb_error_name(transfer->status));
    break;
  case LIBUSB_TRANSFER_COMPLETED:
    uvc_process_status_xfer(devh, transfer);
    break;
  case LIBUSB_TRANSFER_TIMED_OUT:
  case LIBUSB_TRANSFER_STALL:
  case LIBUSB_TRANSFER_OVERFLOW:
    UVC_DEBUG("retrying transfer, status = %s", libusb_error_name(transfer->status));
    break;
  }

  uvc_error_t ret = libusb_submit_transfer(transfer);
  UVC_DEBUG("libusb_submit_transfer() = %d", ret);

  UVC_EXIT_VOID();
}

/** @brief Set a callback function to receive status updates
 *
 * @ingroup device
 */
void uvc_set_status_callback(uvc_device_handle_t *devh,
                             uvc_status_callback_t cb,
                             void *user_ptr) {
  UVC_ENTER();

  devh->status_cb = cb;
  devh->status_user_ptr = user_ptr;

  UVC_EXIT_VOID();
}


/**
 * @brief Get format descriptions for the open device.
 *
 * @note Do not modify the returned structure.
 *
 * @param devh Device handle to an open UVC device
 */
const uvc_format_desc_t *uvc_get_format_descs(uvc_device_handle_t *devh) {
  return devh->info->stream_ifs->format_descs;
}

