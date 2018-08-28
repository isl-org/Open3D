// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#if defined(RS_USE_LIBUVC_BACKEND) && !defined(RS_USE_WMF_BACKEND) && !defined(RS_USE_V4L2_BACKEND)
// UVC support will be provided via libuvc / libusb backend
#elif !defined(RS_USE_LIBUVC_BACKEND) && defined(RS_USE_WMF_BACKEND) && !defined(RS_USE_V4L2_BACKEND)
// UVC support will be provided via Windows Media Foundation / WinUSB backend
#elif !defined(RS_USE_LIBUVC_BACKEND) && !defined(RS_USE_WMF_BACKEND) && defined(RS_USE_V4L2_BACKEND)
// UVC support will be provided via Video 4 Linux 2 / libusb backend
#else
#error No UVC backend selected. Please #define exactly one of RS_USE_LIBUVC_BACKEND, RS_USE_WMF_BACKEND, or RS_USE_V4L2_BACKEND
#endif