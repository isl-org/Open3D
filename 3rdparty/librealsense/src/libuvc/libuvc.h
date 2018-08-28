#ifndef LIBUVC_H
#define LIBUVC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h> // FILE
#include <errno.h>
//#include <libusb-1.0/libusb.h>
#include <libusb.h>
#include "libuvc_config.h"

/** UVC error types, based on libusb errors
 * @ingroup diag
 */
typedef enum uvc_error {
  /** Success (no error) */
  UVC_SUCCESS = 0,
  /** Input/output error */
  UVC_ERROR_IO = -1,
  /** Invalid parameter */
  UVC_ERROR_INVALID_PARAM = -2,
  /** Access denied */
  UVC_ERROR_ACCESS = -3,
  /** No such device */
  UVC_ERROR_NO_DEVICE = -4,
  /** Entity not found */
  UVC_ERROR_NOT_FOUND = -5,
  /** Resource busy */
  UVC_ERROR_BUSY = -6,
  /** Operation timed out */
  UVC_ERROR_TIMEOUT = -7,
  /** Overflow */
  UVC_ERROR_OVERFLOW = -8,
  /** Pipe error */
  UVC_ERROR_PIPE = -9,
  /** System call interrupted */
  UVC_ERROR_INTERRUPTED = -10,
  /** Insufficient memory */
  UVC_ERROR_NO_MEM = -11,
  /** Operation not supported */
  UVC_ERROR_NOT_SUPPORTED = -12,
  /** Device is not UVC-compliant */
  UVC_ERROR_INVALID_DEVICE = -50,
  /** Mode not supported */
  UVC_ERROR_INVALID_MODE = -51,
  /** Resource has a callback (can't use polling and async) */
  UVC_ERROR_CALLBACK_EXISTS = -52,
  /** Undefined error */
  UVC_ERROR_OTHER = -99
} uvc_error_t;

/** VideoStreaming interface descriptor subtype (A.6) */
enum uvc_vs_desc_subtype {
  UVC_VS_UNDEFINED = 0x00,
  UVC_VS_INPUT_HEADER = 0x01,
  UVC_VS_OUTPUT_HEADER = 0x02,
  UVC_VS_STILL_IMAGE_FRAME = 0x03,
  UVC_VS_FORMAT_UNCOMPRESSED = 0x04,
  UVC_VS_FRAME_UNCOMPRESSED = 0x05,
  UVC_VS_FORMAT_MJPEG = 0x06,
  UVC_VS_FRAME_MJPEG = 0x07,
  UVC_VS_FORMAT_MPEG2TS = 0x0a,
  UVC_VS_FORMAT_DV = 0x0c,
  UVC_VS_COLORFORMAT = 0x0d,
  UVC_VS_FORMAT_FRAME_BASED = 0x10,
  UVC_VS_FRAME_FRAME_BASED = 0x11,
  UVC_VS_FORMAT_STREAM_BASED = 0x12
};

struct uvc_format_desc;
struct uvc_frame_desc;

/** Frame descriptor
 *
 * A "frame" is a configuration of a streaming format
 * for a particular image size at one of possibly several
 * available frame rates.
 */
typedef struct uvc_frame_desc {
  struct uvc_format_desc *parent;
  struct uvc_frame_desc *prev, *next;
  /** Type of frame, such as JPEG frame or uncompressed frme */
  enum uvc_vs_desc_subtype bDescriptorSubtype;
  /** Index of the frame within the list of specs available for this format */
  uint8_t bFrameIndex;
  uint8_t bmCapabilities;
  /** Image width */
  uint16_t wWidth;
  /** Image height */
  uint16_t wHeight;
  /** Bitrate of corresponding stream at minimal frame rate */
  uint32_t dwMinBitRate;
  /** Bitrate of corresponding stream at maximal frame rate */
  uint32_t dwMaxBitRate;
  /** Maximum number of bytes for a video frame */
  uint32_t dwMaxVideoFrameBufferSize;
  /** Default frame interval (in 100ns units) */
  uint32_t dwDefaultFrameInterval;
  /** Minimum frame interval for continuous mode (100ns units) */
  uint32_t dwMinFrameInterval;
  /** Maximum frame interval for continuous mode (100ns units) */
  uint32_t dwMaxFrameInterval;
  /** Granularity of frame interval range for continuous mode (100ns) */
  uint32_t dwFrameIntervalStep;
  /** Frame intervals */
  uint8_t bFrameIntervalType;
  /** number of bytes per line */
  uint32_t dwBytesPerLine;
  /** Available frame rates, zero-terminated (in 100ns units) */
  uint32_t *intervals;
} uvc_frame_desc_t;

/** Format descriptor
 *
 * A "format" determines a stream's image type (e.g., raw YUYV or JPEG)
 * and includes many "frame" configurations.
 */
typedef struct uvc_format_desc {
  struct uvc_streaming_interface *parent;
  struct uvc_format_desc *prev, *next;
  /** Type of image stream, such as JPEG or uncompressed. */
  enum uvc_vs_desc_subtype bDescriptorSubtype;
  /** Identifier of this format within the VS interface's format list */
  uint8_t bFormatIndex;
  uint8_t bNumFrameDescriptors;
  /** Format specifier */
  union {
    uint8_t guidFormat[16];
    uint8_t fourccFormat[4];
  };
  /** Format-specific data */
  union {
    /** BPP for uncompressed stream */
    uint8_t bBitsPerPixel;
    /** Flags for JPEG stream */
    uint8_t bmFlags;
  };
  /** Default {uvc_frame_desc} to choose given this format */
  uint8_t bDefaultFrameIndex;
  uint8_t bAspectRatioX;
  uint8_t bAspectRatioY;
  uint8_t bmInterlaceFlags;
  uint8_t bCopyProtect;
  uint8_t bVariableSize;
  /** Available frame specifications for this format */
  struct uvc_frame_desc *frame_descs;
} uvc_format_desc_t;

/** UVC request code (A.8) */
enum uvc_req_code {
  UVC_RC_UNDEFINED = 0x00,
  UVC_SET_CUR = 0x01,
  UVC_GET_CUR = 0x81,
  UVC_GET_MIN = 0x82,
  UVC_GET_MAX = 0x83,
  UVC_GET_RES = 0x84,
  UVC_GET_LEN = 0x85,
  UVC_GET_INFO = 0x86,
  UVC_GET_DEF = 0x87
};

enum uvc_device_power_mode {
  UVC_VC_VIDEO_POWER_MODE_FULL = 0x000b,
  UVC_VC_VIDEO_POWER_MODE_DEVICE_DEPENDENT = 0x001b,
};

/** Camera terminal control selector (A.9.4) */
enum uvc_ct_ctrl_selector {
  UVC_CT_CONTROL_UNDEFINED = 0x00,
  UVC_CT_SCANNING_MODE_CONTROL = 0x01,
  UVC_CT_AE_MODE_CONTROL = 0x02,
  UVC_CT_AE_PRIORITY_CONTROL = 0x03,
  UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL = 0x04,
  UVC_CT_EXPOSURE_TIME_RELATIVE_CONTROL = 0x05,
  UVC_CT_FOCUS_ABSOLUTE_CONTROL = 0x06,
  UVC_CT_FOCUS_RELATIVE_CONTROL = 0x07,
  UVC_CT_FOCUS_AUTO_CONTROL = 0x08,
  UVC_CT_IRIS_ABSOLUTE_CONTROL = 0x09,
  UVC_CT_IRIS_RELATIVE_CONTROL = 0x0a,
  UVC_CT_ZOOM_ABSOLUTE_CONTROL = 0x0b,
  UVC_CT_ZOOM_RELATIVE_CONTROL = 0x0c,
  UVC_CT_PANTILT_ABSOLUTE_CONTROL = 0x0d,
  UVC_CT_PANTILT_RELATIVE_CONTROL = 0x0e,
  UVC_CT_ROLL_ABSOLUTE_CONTROL = 0x0f,
  UVC_CT_ROLL_RELATIVE_CONTROL = 0x10,
  UVC_CT_PRIVACY_CONTROL = 0x11,
  UVC_CT_FOCUS_SIMPLE_CONTROL = 0x12,
  UVC_CT_DIGITAL_WINDOW_CONTROL = 0x13,
  UVC_CT_REGION_OF_INTEREST_CONTROL = 0x14
};

/** Processing unit control selector (A.9.5) */
enum uvc_pu_ctrl_selector {
  UVC_PU_CONTROL_UNDEFINED = 0x00,
  UVC_PU_BACKLIGHT_COMPENSATION_CONTROL = 0x01,
  UVC_PU_BRIGHTNESS_CONTROL = 0x02,
  UVC_PU_CONTRAST_CONTROL = 0x03,
  UVC_PU_GAIN_CONTROL = 0x04,
  UVC_PU_POWER_LINE_FREQUENCY_CONTROL = 0x05,
  UVC_PU_HUE_CONTROL = 0x06,
  UVC_PU_SATURATION_CONTROL = 0x07,
  UVC_PU_SHARPNESS_CONTROL = 0x08,
  UVC_PU_GAMMA_CONTROL = 0x09,
  UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL = 0x0a,
  UVC_PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL = 0x0b,
  UVC_PU_WHITE_BALANCE_COMPONENT_CONTROL = 0x0c,
  UVC_PU_WHITE_BALANCE_COMPONENT_AUTO_CONTROL = 0x0d,
  UVC_PU_DIGITAL_MULTIPLIER_CONTROL = 0x0e,
  UVC_PU_DIGITAL_MULTIPLIER_LIMIT_CONTROL = 0x0f,
  UVC_PU_HUE_AUTO_CONTROL = 0x10,
  UVC_PU_ANALOG_VIDEO_STANDARD_CONTROL = 0x11,
  UVC_PU_ANALOG_LOCK_STATUS_CONTROL = 0x12,
  UVC_PU_CONTRAST_AUTO_CONTROL = 0x13
};

/** USB terminal type (B.1) */
enum uvc_term_type {
  UVC_TT_VENDOR_SPECIFIC = 0x0100,
  UVC_TT_STREAMING = 0x0101
};

/** Input terminal type (B.2) */
enum uvc_it_type {
  UVC_ITT_VENDOR_SPECIFIC = 0x0200,
  UVC_ITT_CAMERA = 0x0201,
  UVC_ITT_MEDIA_TRANSPORT_INPUT = 0x0202
};

/** Output terminal type (B.3) */
enum uvc_ot_type {
  UVC_OTT_VENDOR_SPECIFIC = 0x0300,
  UVC_OTT_DISPLAY = 0x0301,
  UVC_OTT_MEDIA_TRANSPORT_OUTPUT = 0x0302
};

/** External terminal type (B.4) */
enum uvc_et_type {
  UVC_EXTERNAL_VENDOR_SPECIFIC = 0x0400,
  UVC_COMPOSITE_CONNECTOR = 0x0401,
  UVC_SVIDEO_CONNECTOR = 0x0402,
  UVC_COMPONENT_CONNECTOR = 0x0403
};

/** Context, equivalent to libusb's contexts.
 *
 * May either own a libusb context or use one that's already made.
 *
 * Always create these with uvc_get_context.
 */
struct uvc_context;
typedef struct uvc_context uvc_context_t;

/** UVC device.
 *
 * Get this from uvc_get_device_list() or uvc_find_device().
 */
struct uvc_device;
typedef struct uvc_device uvc_device_t;

/** Handle on an open UVC device.
 *
 * Get one of these from uvc_open(). Once you uvc_close()
 * it, it's no longer valid.
 */
struct uvc_device_handle;
typedef struct uvc_device_handle uvc_device_handle_t;

/** Handle on an open UVC stream.
 *
 * Get one of these from uvc_stream_open*().
 * Once you uvc_stream_close() it, it will no longer be valid.
 */
struct uvc_stream_handle;
typedef struct uvc_stream_handle uvc_stream_handle_t;

/** Representation of the interface that brings data into the UVC device */
typedef struct uvc_input_terminal {
  struct uvc_input_terminal *prev, *next;
  /** Index of the terminal within the device */
  uint8_t bTerminalID;
  /** Type of terminal (e.g., camera) */
  enum uvc_it_type wTerminalType;
  uint16_t wObjectiveFocalLengthMin;
  uint16_t wObjectiveFocalLengthMax;
  uint16_t wOcularFocalLength;
  /** Camera controls (meaning of bits given in {uvc_ct_ctrl_selector}) */
  uint64_t bmControls;
} uvc_input_terminal_t;

typedef struct uvc_output_terminal {
  struct uvc_output_terminal *prev, *next;
  /** @todo */
} uvc_output_terminal_t;

/** Represents post-capture processing functions */
typedef struct uvc_processing_unit {
  struct uvc_processing_unit *prev, *next;
  /** Index of the processing unit within the device */
  uint8_t bUnitID;
  /** Index of the terminal from which the device accepts images */
  uint8_t bSourceID;
  /** Processing controls (meaning of bits given in {uvc_pu_ctrl_selector}) */
  uint64_t bmControls;
} uvc_processing_unit_t;

/** Custom processing or camera-control functions */
typedef struct uvc_extension_unit {
  struct uvc_extension_unit *prev, *next;
  /** Index of the extension unit within the device */
  uint8_t bUnitID;
  /** GUID identifying the extension unit */
  uint8_t guidExtensionCode[16];
  /** Bitmap of available controls (manufacturer-dependent) */
  uint64_t bmControls;
} uvc_extension_unit_t;

enum uvc_status_class {
  UVC_STATUS_CLASS_CONTROL = 0x10,
  UVC_STATUS_CLASS_CONTROL_CAMERA = 0x11,
  UVC_STATUS_CLASS_CONTROL_PROCESSING = 0x12,
};

enum uvc_status_attribute {
  UVC_STATUS_ATTRIBUTE_VALUE_CHANGE = 0x00,
  UVC_STATUS_ATTRIBUTE_INFO_CHANGE = 0x01,
  UVC_STATUS_ATTRIBUTE_FAILURE_CHANGE = 0x02,
  UVC_STATUS_ATTRIBUTE_UNKNOWN = 0xff
};

/** A callback function to accept status updates
 * @ingroup device
 */
typedef void(uvc_status_callback_t)(enum uvc_status_class status_class,
                                    int event,
                                    int selector,
                                    enum uvc_status_attribute status_attribute,
                                    void *data, size_t data_len,
                                    void *user_ptr);

/** Structure representing a UVC device descriptor.
 *
 * (This isn't a standard structure.)
 */
typedef struct uvc_device_descriptor {
  /** Vendor ID */
  uint16_t idVendor;
  /** Product ID */
  uint16_t idProduct;
  /** UVC compliance level, e.g. 0x0100 (1.0), 0x0110 */
  uint16_t bcdUVC;
  /** Serial number (null if unavailable) */
  const char *serialNumber;
  /** Device-reported manufacturer name (or null) */
  const char *manufacturer;
  /** Device-reporter product name (or null) */
  const char *product;
} uvc_device_descriptor_t;

/** An image frame received from the UVC device
 * @ingroup streaming
 */
typedef struct uvc_frame {
  /** Image data for this frame */
  void *data;
  /** Size of image data buffer */
  size_t data_bytes;
  /** Width of image in pixels */
  uint32_t width;
  /** Height of image in pixels */
  uint32_t height;
  /** Pixel data format */
  uint32_t fourcc;
  /** Number of bytes per horizontal line (undefined for compressed format) */
  size_t step;
  /** Frame number (may skip, but is strictly monotonically increasing) */
  uint32_t sequence;
  /** Estimate of system time when the device started capturing the image */
  struct timeval capture_time;
  /** Handle on the device that produced the image.
   * @warning You must not call any uvc_* functions during a callback. */
  uvc_device_handle_t *source;
  /** Is the data buffer owned by the library?
   * If 1, the data buffer can be arbitrarily reallocated by frame conversion
   * functions.
   * If 0, the data buffer will not be reallocated or freed by the library.
   * Set this field to zero if you are supplying the buffer.
   */
  uint8_t library_owns_data;
} uvc_frame_t;

/** A callback function to handle incoming assembled UVC frames
 * @ingroup streaming
 */
typedef void(uvc_frame_callback_t)(struct uvc_frame *frame, void *user_ptr);

/** Streaming mode, includes all information needed to select stream
 * @ingroup streaming
 */
typedef struct uvc_stream_ctrl {
  uint16_t bmHint;
  uint8_t bFormatIndex;
  uint8_t bFrameIndex;
  uint32_t dwFrameInterval;
  uint16_t wKeyFrameRate;
  uint16_t wPFrameRate;
  uint16_t wCompQuality;
  uint16_t wCompWindowSize;
  uint16_t wDelay;
  uint32_t dwMaxVideoFrameSize;
  uint32_t dwMaxPayloadTransferSize;
  uint32_t dwClockFrequency;
  uint8_t bmFramingInfo;
  uint8_t bPreferredVersion;
  uint8_t bMinVersion;
  uint8_t bMaxVersion;
  uint8_t bInterfaceNumber;
  uvc_stream_handle_t *handle;
} uvc_stream_ctrl_t;

uvc_error_t uvc_init(uvc_context_t **ctx, struct libusb_context *usb_ctx);
void uvc_exit(uvc_context_t *ctx);

uvc_error_t uvc_get_device_list(
    uvc_context_t *ctx,
    uvc_device_t ***list);
void uvc_free_device_list(uvc_device_t **list, uint8_t unref_devices);

uvc_error_t uvc_get_device_descriptor(
    uvc_device_t *dev,
    uvc_device_descriptor_t **desc);
void uvc_free_device_descriptor(
    uvc_device_descriptor_t *desc);

uint8_t uvc_get_bus_number(uvc_device_t *dev);
uint8_t uvc_get_device_address(uvc_device_t *dev);

uvc_error_t uvc_find_device(
    uvc_context_t *ctx,
    uvc_device_t **dev,
    int vid, int pid, const char *sn);

uvc_error_t uvc_open(
    uvc_device_t *dev,
    uvc_device_handle_t **devh);

uvc_error_t uvc_open2(
    uvc_device_t *dev,
    uvc_device_handle_t **devh,
    int camera_number);

void uvc_close(uvc_device_handle_t *devh);

uvc_device_t *uvc_get_device(uvc_device_handle_t *devh);
libusb_device_handle *uvc_get_libusb_handle(uvc_device_handle_t *devh);

void uvc_ref_device(uvc_device_t *dev);
void uvc_unref_device(uvc_device_t *dev);

void uvc_set_status_callback(uvc_device_handle_t *devh,
                             uvc_status_callback_t cb,
                             void *user_ptr);

const uvc_input_terminal_t *uvc_get_input_terminals(uvc_device_handle_t *devh);
const uvc_output_terminal_t *uvc_get_output_terminals(uvc_device_handle_t *devh);
const uvc_processing_unit_t *uvc_get_processing_units(uvc_device_handle_t *devh);
const uvc_extension_unit_t *uvc_get_extension_units(uvc_device_handle_t *devh);

uvc_error_t uvc_get_stream_ctrl_format_size(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl,
    uint32_t fourcc,
    int width, int height,
    int fps
    );

const uvc_format_desc_t *uvc_get_format_descs(uvc_device_handle_t* );

uvc_error_t uvc_probe_stream_ctrl(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl);

uvc_error_t uvc_start_streaming(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl,
    uvc_frame_callback_t *cb,
    void *user_ptr,
    uint8_t flags,
    int num_transfer_buffers);

uvc_error_t uvc_start_iso_streaming(
    uvc_device_handle_t *devh,
    uvc_stream_ctrl_t *ctrl,
    uvc_frame_callback_t *cb,
    void *user_ptr);

void uvc_stop_streaming(uvc_device_handle_t *devh);

uvc_error_t uvc_stream_open_ctrl(uvc_device_handle_t *devh, uvc_stream_handle_t **strmh, uvc_stream_ctrl_t *ctrl);
uvc_error_t uvc_stream_ctrl(uvc_stream_handle_t *strmh, uvc_stream_ctrl_t *ctrl);
uvc_error_t uvc_stream_start(uvc_stream_handle_t *strmh,
    uvc_frame_callback_t *cb,
    void *user_ptr,
    uint8_t flags,
    int num_transfer_buffers);
uvc_error_t uvc_stream_start_iso(uvc_stream_handle_t *strmh,
    uvc_frame_callback_t *cb,
    void *user_ptr);
uvc_error_t uvc_stream_get_frame(
    uvc_stream_handle_t *strmh,
    uvc_frame_t **frame,
    int32_t timeout_us
);
uvc_error_t uvc_stream_stop(uvc_stream_handle_t *strmh);
void uvc_stream_close(uvc_stream_handle_t *strmh);

int uvc_get_ctrl_len(uvc_device_handle_t *devh, uint8_t unit, uint8_t ctrl);
int uvc_get_ctrl(uvc_device_handle_t *devh, uint8_t unit, uint8_t ctrl, void *data, int len, enum uvc_req_code req_code);
int uvc_set_ctrl(uvc_device_handle_t *devh, uint8_t unit, uint8_t ctrl, void *data, int len);

uvc_error_t uvc_get_power_mode(uvc_device_handle_t *devh, enum uvc_device_power_mode *mode, enum uvc_req_code req_code);
uvc_error_t uvc_set_power_mode(uvc_device_handle_t *devh, enum uvc_device_power_mode mode);

/* AUTO-GENERATED control accessors! Update them with the output of `ctrl-gen.py decl`. */
uvc_error_t uvc_get_scanning_mode(uvc_device_handle_t *devh, uint8_t* mode, enum uvc_req_code req_code);
uvc_error_t uvc_set_scanning_mode(uvc_device_handle_t *devh, uint8_t mode);

uvc_error_t uvc_get_ae_mode(uvc_device_handle_t *devh, uint8_t* mode, enum uvc_req_code req_code);
uvc_error_t uvc_set_ae_mode(uvc_device_handle_t *devh, uint8_t mode);

uvc_error_t uvc_get_ae_priority(uvc_device_handle_t *devh, uint8_t* priority, enum uvc_req_code req_code);
uvc_error_t uvc_set_ae_priority(uvc_device_handle_t *devh, uint8_t priority);

uvc_error_t uvc_get_exposure_abs(uvc_device_handle_t *devh, uint32_t* time, enum uvc_req_code req_code);
uvc_error_t uvc_set_exposure_abs(uvc_device_handle_t *devh, uint32_t time);

uvc_error_t uvc_get_exposure_rel(uvc_device_handle_t *devh, int8_t* step, enum uvc_req_code req_code);
uvc_error_t uvc_set_exposure_rel(uvc_device_handle_t *devh, int8_t step);

uvc_error_t uvc_get_focus_abs(uvc_device_handle_t *devh, uint16_t* focus, enum uvc_req_code req_code);
uvc_error_t uvc_set_focus_abs(uvc_device_handle_t *devh, uint16_t focus);

uvc_error_t uvc_get_focus_rel(uvc_device_handle_t *devh, int8_t* focus_rel, uint8_t* speed, enum uvc_req_code req_code);
uvc_error_t uvc_set_focus_rel(uvc_device_handle_t *devh, int8_t focus_rel, uint8_t speed);

uvc_error_t uvc_get_focus_simple_range(uvc_device_handle_t *devh, uint8_t* focus, enum uvc_req_code req_code);
uvc_error_t uvc_set_focus_simple_range(uvc_device_handle_t *devh, uint8_t focus);

uvc_error_t uvc_get_focus_auto(uvc_device_handle_t *devh, uint8_t* state, enum uvc_req_code req_code);
uvc_error_t uvc_set_focus_auto(uvc_device_handle_t *devh, uint8_t state);

uvc_error_t uvc_get_iris_abs(uvc_device_handle_t *devh, uint16_t* iris, enum uvc_req_code req_code);
uvc_error_t uvc_set_iris_abs(uvc_device_handle_t *devh, uint16_t iris);

uvc_error_t uvc_get_iris_rel(uvc_device_handle_t *devh, uint8_t* iris_rel, enum uvc_req_code req_code);
uvc_error_t uvc_set_iris_rel(uvc_device_handle_t *devh, uint8_t iris_rel);

uvc_error_t uvc_get_zoom_abs(uvc_device_handle_t *devh, uint16_t* focal_length, enum uvc_req_code req_code);
uvc_error_t uvc_set_zoom_abs(uvc_device_handle_t *devh, uint16_t focal_length);

uvc_error_t uvc_get_zoom_rel(uvc_device_handle_t *devh, int8_t* zoom_rel, uint8_t* digital_zoom, uint8_t* speed, enum uvc_req_code req_code);
uvc_error_t uvc_set_zoom_rel(uvc_device_handle_t *devh, int8_t zoom_rel, uint8_t digital_zoom, uint8_t speed);

uvc_error_t uvc_get_pantilt_abs(uvc_device_handle_t *devh, int32_t* pan, int32_t* tilt, enum uvc_req_code req_code);
uvc_error_t uvc_set_pantilt_abs(uvc_device_handle_t *devh, int32_t pan, int32_t tilt);

uvc_error_t uvc_get_pantilt_rel(uvc_device_handle_t *devh, int8_t* pan_rel, uint8_t* pan_speed, int8_t* tilt_rel, uint8_t* tilt_speed, enum uvc_req_code req_code);
uvc_error_t uvc_set_pantilt_rel(uvc_device_handle_t *devh, int8_t pan_rel, uint8_t pan_speed, int8_t tilt_rel, uint8_t tilt_speed);

uvc_error_t uvc_get_roll_abs(uvc_device_handle_t *devh, int16_t* roll, enum uvc_req_code req_code);
uvc_error_t uvc_set_roll_abs(uvc_device_handle_t *devh, int16_t roll);

uvc_error_t uvc_get_roll_rel(uvc_device_handle_t *devh, int8_t* roll_rel, uint8_t* speed, enum uvc_req_code req_code);
uvc_error_t uvc_set_roll_rel(uvc_device_handle_t *devh, int8_t roll_rel, uint8_t speed);

uvc_error_t uvc_get_privacy(uvc_device_handle_t *devh, uint8_t* privacy, enum uvc_req_code req_code);
uvc_error_t uvc_set_privacy(uvc_device_handle_t *devh, uint8_t privacy);

uvc_error_t uvc_get_digital_window(uvc_device_handle_t *devh, uint16_t* window_top, uint16_t* window_left, uint16_t* window_bottom, uint16_t* window_right, uint16_t* num_steps, uint16_t* num_steps_units, enum uvc_req_code req_code);
uvc_error_t uvc_set_digital_window(uvc_device_handle_t *devh, uint16_t window_top, uint16_t window_left, uint16_t window_bottom, uint16_t window_right, uint16_t num_steps, uint16_t num_steps_units);

uvc_error_t uvc_get_digital_roi(uvc_device_handle_t *devh, uint16_t* roi_top, uint16_t* roi_left, uint16_t* roi_bottom, uint16_t* roi_right, uint16_t* auto_controls, enum uvc_req_code req_code);
uvc_error_t uvc_set_digital_roi(uvc_device_handle_t *devh, uint16_t roi_top, uint16_t roi_left, uint16_t roi_bottom, uint16_t roi_right, uint16_t auto_controls);

uvc_error_t uvc_get_backlight_compensation(uvc_device_handle_t *devh, uint16_t* backlight_compensation, enum uvc_req_code req_code);
uvc_error_t uvc_set_backlight_compensation(uvc_device_handle_t *devh, uint16_t backlight_compensation);

uvc_error_t uvc_get_brightness(uvc_device_handle_t *devh, int16_t* brightness, enum uvc_req_code req_code);
uvc_error_t uvc_set_brightness(uvc_device_handle_t *devh, int16_t brightness);

uvc_error_t uvc_get_contrast(uvc_device_handle_t *devh, uint16_t* contrast, enum uvc_req_code req_code);
uvc_error_t uvc_set_contrast(uvc_device_handle_t *devh, uint16_t contrast);

uvc_error_t uvc_get_contrast_auto(uvc_device_handle_t *devh, uint8_t* contrast_auto, enum uvc_req_code req_code);
uvc_error_t uvc_set_contrast_auto(uvc_device_handle_t *devh, uint8_t contrast_auto);

uvc_error_t uvc_get_gain(uvc_device_handle_t *devh, uint16_t* gain, enum uvc_req_code req_code);
uvc_error_t uvc_set_gain(uvc_device_handle_t *devh, uint16_t gain);

uvc_error_t uvc_get_power_line_frequency(uvc_device_handle_t *devh, uint8_t* power_line_frequency, enum uvc_req_code req_code);
uvc_error_t uvc_set_power_line_frequency(uvc_device_handle_t *devh, uint8_t power_line_frequency);

uvc_error_t uvc_get_hue(uvc_device_handle_t *devh, int16_t* hue, enum uvc_req_code req_code);
uvc_error_t uvc_set_hue(uvc_device_handle_t *devh, int16_t hue);

uvc_error_t uvc_get_hue_auto(uvc_device_handle_t *devh, uint8_t* hue_auto, enum uvc_req_code req_code);
uvc_error_t uvc_set_hue_auto(uvc_device_handle_t *devh, uint8_t hue_auto);

uvc_error_t uvc_get_saturation(uvc_device_handle_t *devh, uint16_t* saturation, enum uvc_req_code req_code);
uvc_error_t uvc_set_saturation(uvc_device_handle_t *devh, uint16_t saturation);

uvc_error_t uvc_get_sharpness(uvc_device_handle_t *devh, uint16_t* sharpness, enum uvc_req_code req_code);
uvc_error_t uvc_set_sharpness(uvc_device_handle_t *devh, uint16_t sharpness);

uvc_error_t uvc_get_gamma(uvc_device_handle_t *devh, uint16_t* gamma, enum uvc_req_code req_code);
uvc_error_t uvc_set_gamma(uvc_device_handle_t *devh, uint16_t gamma);

uvc_error_t uvc_get_white_balance_temperature(uvc_device_handle_t *devh, uint16_t* temperature, enum uvc_req_code req_code);
uvc_error_t uvc_set_white_balance_temperature(uvc_device_handle_t *devh, uint16_t temperature);

uvc_error_t uvc_get_white_balance_temperature_auto(uvc_device_handle_t *devh, uint8_t* temperature_auto, enum uvc_req_code req_code);
uvc_error_t uvc_set_white_balance_temperature_auto(uvc_device_handle_t *devh, uint8_t temperature_auto);

uvc_error_t uvc_get_white_balance_component(uvc_device_handle_t *devh, uint16_t* blue, uint16_t* red, enum uvc_req_code req_code);
uvc_error_t uvc_set_white_balance_component(uvc_device_handle_t *devh, uint16_t blue, uint16_t red);

uvc_error_t uvc_get_white_balance_component_auto(uvc_device_handle_t *devh, uint8_t* white_balance_component_auto, enum uvc_req_code req_code);
uvc_error_t uvc_set_white_balance_component_auto(uvc_device_handle_t *devh, uint8_t white_balance_component_auto);

uvc_error_t uvc_get_digital_multiplier(uvc_device_handle_t *devh, uint16_t* multiplier_step, enum uvc_req_code req_code);
uvc_error_t uvc_set_digital_multiplier(uvc_device_handle_t *devh, uint16_t multiplier_step);

uvc_error_t uvc_get_digital_multiplier_limit(uvc_device_handle_t *devh, uint16_t* multiplier_step, enum uvc_req_code req_code);
uvc_error_t uvc_set_digital_multiplier_limit(uvc_device_handle_t *devh, uint16_t multiplier_step);

uvc_error_t uvc_get_analog_video_standard(uvc_device_handle_t *devh, uint8_t* video_standard, enum uvc_req_code req_code);
uvc_error_t uvc_set_analog_video_standard(uvc_device_handle_t *devh, uint8_t video_standard);

uvc_error_t uvc_get_analog_video_lock_status(uvc_device_handle_t *devh, uint8_t* status, enum uvc_req_code req_code);
uvc_error_t uvc_set_analog_video_lock_status(uvc_device_handle_t *devh, uint8_t status);

uvc_error_t uvc_get_input_select(uvc_device_handle_t *devh, uint8_t* selector, enum uvc_req_code req_code);
uvc_error_t uvc_set_input_select(uvc_device_handle_t *devh, uint8_t selector);
/* end AUTO-GENERATED control accessors */

void uvc_perror(uvc_error_t err, const char *msg);
const char* uvc_strerror(uvc_error_t err);
void uvc_print_diag(uvc_device_handle_t *devh, FILE *stream);
void uvc_print_stream_ctrl(uvc_stream_ctrl_t *ctrl, FILE *stream);

uvc_frame_t *uvc_allocate_frame(size_t data_bytes);
void uvc_free_frame(uvc_frame_t *frame);

uvc_error_t uvc_duplicate_frame(uvc_frame_t *in, uvc_frame_t *out);

#ifdef __cplusplus
}
#endif

#endif // !def(LIBUVC_H)
