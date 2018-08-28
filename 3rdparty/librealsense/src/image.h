// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once
#ifndef LIBREALSENSE_IMAGE_H
#define LIBREALSENSE_IMAGE_H

#include "types.h"

namespace rsimpl
{
    size_t           get_image_size                 (int width, int height, rs_format format);

    void             deproject_z                    (float * points, const rs_intrinsics & z_intrin, const uint16_t * z_pixels, float z_scale);
    void             deproject_disparity            (float * points, const rs_intrinsics & disparity_intrin, const uint16_t * disparity_pixels, float disparity_scale);

    void             align_z_to_other               (byte * z_aligned_to_other, const uint16_t * z_pixels, float z_scale, const rs_intrinsics & z_intrin, 
                                                     const rs_extrinsics & z_to_other, const rs_intrinsics & other_intrin);
    void             align_disparity_to_other       (byte * disparity_aligned_to_other, const uint16_t * disparity_pixels, float disparity_scale, const rs_intrinsics & disparity_intrin, 
                                                     const rs_extrinsics & disparity_to_other, const rs_intrinsics & other_intrin);
    void             align_other_to_z               (byte * other_aligned_to_z, const uint16_t * z_pixels, float z_scale, const rs_intrinsics & z_intrin, 
                                                     const rs_extrinsics & z_to_other, const rs_intrinsics & other_intrin, const byte * other_pixels, rs_format other_format);
    void             align_other_to_disparity       (byte * other_aligned_to_disparity, const uint16_t * disparity_pixels, float disparity_scale, const rs_intrinsics & disparity_intrin, 
                                                     const rs_extrinsics & disparity_to_other, const rs_intrinsics & other_intrin, const byte * other_pixels, rs_format other_format);

    std::vector<int> compute_rectification_table    (const rs_intrinsics & rect_intrin, const rs_extrinsics & rect_to_unrect, const rs_intrinsics & unrect_intrin);
    void             rectify_image                  (byte * rect_pixels, const std::vector<int> & rectification_table, const byte * unrect_pixels, rs_format format);

    extern const native_pixel_format pf_rw10;       // Four 10 bit luminance values in one 40 bit macropixel
    extern const native_pixel_format pf_yuy2;       // Y0 U Y1 V ordered chroma subsampled macropixel
    extern const native_pixel_format pf_y8;         // 8 bit (left) IR image
    extern const native_pixel_format pf_y8i;        // 8 bits left IR + 8 bits right IR per pixel
    extern const native_pixel_format pf_y16;        // 16 bit (left) IR image
    extern const native_pixel_format pf_y12i;       // 12 bits left IR + 12 bits right IR per pixel
    extern const native_pixel_format pf_z16;        // 16 bit Z image
    extern const native_pixel_format pf_invz;       // 16 bit Z image
    extern const native_pixel_format pf_f200_invi;  // 8-bit IR image
    extern const native_pixel_format pf_f200_inzi;  // 16-bit Z + 8 bit IR per pixel
    extern const native_pixel_format pf_sr300_invi; // 16-bit IR image
    extern const native_pixel_format pf_sr300_inzi; // Planar 16-bit IR image followed by 16-bit Z image
}

#endif
