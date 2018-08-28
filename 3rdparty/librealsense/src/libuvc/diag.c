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
 * @defgroup diag Diagnostics
 * @brief Interpretation of devices, error codes and negotiated stream parameters
 */

#include "libuvc.h"
#include "libuvc_internal.h"

/** @internal */
typedef struct _uvc_error_msg {
  uvc_error_t err;
  const char *msg;
} _uvc_error_msg_t;

static const _uvc_error_msg_t uvc_error_msgs[] = {
  {UVC_SUCCESS, "Success"},
  {UVC_ERROR_IO, "I/O error"},
  {UVC_ERROR_INVALID_PARAM, "Invalid parameter"},
  {UVC_ERROR_ACCESS, "Access denied"},
  {UVC_ERROR_NO_DEVICE, "No such device"},
  {UVC_ERROR_NOT_FOUND, "Not found"},
  {UVC_ERROR_BUSY, "Busy"},
  {UVC_ERROR_TIMEOUT, "Timeout"},
  {UVC_ERROR_OVERFLOW, "Overflow"},
  {UVC_ERROR_PIPE, "Pipe"},
  {UVC_ERROR_INTERRUPTED, "Interrupted"},
  {UVC_ERROR_NO_MEM, "Out of memory"},
  {UVC_ERROR_NOT_SUPPORTED, "Not supported"},
  {UVC_ERROR_INVALID_DEVICE, "Invalid device"},
  {UVC_ERROR_INVALID_MODE, "Invalid mode"},
  {UVC_ERROR_CALLBACK_EXISTS, "Callback exists"}
};

/** @brief Print a message explaining an error in the UVC driver
 * @ingroup diag
 *
 * @param err UVC error code
 * @param msg Optional custom message, prepended to output
 */
void uvc_perror(uvc_error_t err, const char *msg) {
  if (msg && *msg) {
    fputs(msg, stderr);
    fputs(": ", stderr);
  }

  fprintf(stderr, "%s (%d)\n", uvc_strerror(err), err);
}

/** @brief Return a string explaining an error in the UVC driver
 * @ingroup diag
 *
 * @param err UVC error code
 * @return error message
 */
const char* uvc_strerror(uvc_error_t err) {
  size_t idx;

  for (idx = 0; idx < sizeof(uvc_error_msgs) / sizeof(*uvc_error_msgs); ++idx) {
    if (uvc_error_msgs[idx].err == err) {
      return uvc_error_msgs[idx].msg;
    }
  }

  return "Unknown error";
}

/** @brief Print the values in a stream control block
 * @ingroup diag
 *
 * @param devh UVC device
 * @param stream Output stream (stderr if NULL)
 */
void uvc_print_stream_ctrl(uvc_stream_ctrl_t *ctrl, FILE *stream) {
  if (stream == NULL)
    stream = stderr;

  fprintf(stream, "bmHint: %04x\n", ctrl->bmHint);
  fprintf(stream, "bFormatIndex: %d\n", ctrl->bFormatIndex);
  fprintf(stream, "bFrameIndex: %d\n", ctrl->bFrameIndex);
  fprintf(stream, "dwFrameInterval: %u\n", ctrl->dwFrameInterval);
  fprintf(stream, "wKeyFrameRate: %d\n", ctrl->wKeyFrameRate);
  fprintf(stream, "wPFrameRate: %d\n", ctrl->wPFrameRate);
  fprintf(stream, "wCompQuality: %d\n", ctrl->wCompQuality);
  fprintf(stream, "wCompWindowSize: %d\n", ctrl->wCompWindowSize);
  fprintf(stream, "wDelay: %d\n", ctrl->wDelay);
  fprintf(stream, "dwMaxVideoFrameSize: %u\n", ctrl->dwMaxVideoFrameSize);
  fprintf(stream, "dwMaxPayloadTransferSize: %u\n", ctrl->dwMaxPayloadTransferSize);
  fprintf(stream, "bInterfaceNumber: %d\n", ctrl->bInterfaceNumber);
}

static const char *_uvc_name_for_format_subtype(uint8_t subtype) {
  switch (subtype) {
  case UVC_VS_FORMAT_UNCOMPRESSED:
    return "UncompressedFormat";
  case UVC_VS_FORMAT_MJPEG:
    return "MJPEGFormat";
  case UVC_VS_FORMAT_FRAME_BASED:
    return "FrameFormat";
  default:
    return "Unknown";
  }
}

/** @brief Print camera capabilities and configuration.
 * @ingroup diag
 *
 * @param devh UVC device
 * @param stream Output stream (stderr if NULL)
 */
void uvc_print_diag(uvc_device_handle_t *devh, FILE *stream) {
  if (stream == NULL)
    stream = stderr;

  if (devh->info->ctrl_if.bcdUVC) {
    uvc_streaming_interface_t *stream_if;
    int stream_idx = 0;

    uvc_device_descriptor_t *desc;
    uvc_get_device_descriptor(devh->dev, &desc);

    fprintf(stream, "DEVICE CONFIGURATION (%04x:%04x/%s) ---\n",
        desc->idVendor, desc->idProduct,
        desc->serialNumber ? desc->serialNumber : "[none]");

    uvc_free_device_descriptor(desc);

    fprintf(stream, "Status: %s\n", devh->streams ? "streaming" : "idle");

    fprintf(stream, "VideoControl:\n"
        "\tbcdUVC: 0x%04x\n",
        devh->info->ctrl_if.bcdUVC);

    DL_FOREACH(devh->info->stream_ifs, stream_if) {
      uvc_format_desc_t *fmt_desc;

      ++stream_idx;

      fprintf(stream, "VideoStreaming(%d):\n"
          "\tbEndpointAddress: %d\n\tFormats:\n",
          stream_idx, stream_if->bEndpointAddress);

      DL_FOREACH(stream_if->format_descs, fmt_desc) {
        uvc_frame_desc_t *frame_desc;
        int i;

        switch (fmt_desc->bDescriptorSubtype) {
          case UVC_VS_FORMAT_UNCOMPRESSED:
          case UVC_VS_FORMAT_MJPEG:
          case UVC_VS_FORMAT_FRAME_BASED:
            fprintf(stream,
                "\t\%s(%d)\n"
                "\t\t  bits per pixel: %d\n"
                "\t\t  GUID: ",
                _uvc_name_for_format_subtype(fmt_desc->bDescriptorSubtype),
                fmt_desc->bFormatIndex,
                fmt_desc->bBitsPerPixel);

            for (i = 0; i < 16; ++i)
              fprintf(stream, "%02x", fmt_desc->guidFormat[i]);

            fprintf(stream, " (%4s)\n", fmt_desc->fourccFormat );

            fprintf(stream,
                "\t\t  default frame: %d\n"
                "\t\t  aspect ratio: %dx%d\n"
                "\t\t  interlace flags: %02x\n"
                "\t\t  copy protect: %02x\n",
                fmt_desc->bDefaultFrameIndex,
                fmt_desc->bAspectRatioX,
                fmt_desc->bAspectRatioY,
                fmt_desc->bmInterlaceFlags,
                fmt_desc->bCopyProtect);

            DL_FOREACH(fmt_desc->frame_descs, frame_desc) {
              uint32_t *interval_ptr;

              fprintf(stream,
                  "\t\t\tFrameDescriptor(%d)\n"
                  "\t\t\t  capabilities: %02x\n"
                  "\t\t\t  size: %dx%d\n"
                  "\t\t\t  bit rate: %d-%d\n"
                  "\t\t\t  max frame size: %d\n"
                  "\t\t\t  default interval: 1/%d\n",
                  frame_desc->bFrameIndex,
                  frame_desc->bmCapabilities,
                  frame_desc->wWidth,
                  frame_desc->wHeight,
                  frame_desc->dwMinBitRate,
                  frame_desc->dwMaxBitRate,
                  frame_desc->dwMaxVideoFrameBufferSize,
                  10000000 / frame_desc->dwDefaultFrameInterval);
              if (frame_desc->intervals) {
                for (interval_ptr = frame_desc->intervals;
                     *interval_ptr;
                     ++interval_ptr) {
                  fprintf(stream,
                      "\t\t\t  interval[%d]: 1/%d\n",
              (int) (interval_ptr - frame_desc->intervals),
              10000000 / *interval_ptr);
                }
              } else {
                fprintf(stream,
                    "\t\t\t  min interval[%d] = 1/%d\n"
                    "\t\t\t  max interval[%d] = 1/%d\n",
                    frame_desc->dwMinFrameInterval,
                    10000000 / frame_desc->dwMinFrameInterval,
                    frame_desc->dwMaxFrameInterval,
                    10000000 / frame_desc->dwMaxFrameInterval);
                if (frame_desc->dwFrameIntervalStep)
                  fprintf(stream,
                      "\t\t\t  interval step[%d] = 1/%d\n",
                      frame_desc->dwFrameIntervalStep,
                      10000000 / frame_desc->dwFrameIntervalStep);
              }
            }
            break;
          default:
            fprintf(stream, "\t-UnknownFormat (%d)\n",
                fmt_desc->bDescriptorSubtype );
        }
      }
    }

    fprintf(stream, "END DEVICE CONFIGURATION\n");
  } else {
    fprintf(stream, "uvc_print_diag: Device not configured!\n");
  }
}

