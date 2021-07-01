// -*- C++ -*-
//===-- utils.h -----------------------------------------------------------===//
//
// Copyright (C) 2017-2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include <vector>
#include <algorithm>

class image {
public:
    union pixel {
        std::uint8_t bgra[4];
        std::uint32_t value;
        pixel() {}
        template <typename T> pixel(T b, T g, T r) {
            bgra[0] = (std::uint8_t)b, bgra[1] = (std::uint8_t)g, bgra[2] = (std::uint8_t)r, bgra[3] = 0;
        }
    };
public:
    image(int w = 1920, int h = 1080);

    int width() const { return my_width; }
    int height() const { return my_height; }

    void write(const char* fname) const;
    void fill(std::uint8_t r, std::uint8_t g, std::uint8_t b, int x = -1, int y = -1);

    template <typename F>
    void fill(F f) {

        if(my_data.empty())
            reset(my_width, my_height);

        int i = -1;
        int w = this->my_width;
        std::for_each(my_data.begin(), my_data.end(), [&i, w, f](image::pixel& p) {
            ++i;
            int x = i / w, y = i % w;
            auto val = f(x, y);
            if(val > 255)
                val = 255;
            p = image::pixel(val, val, val);
        });
    }

    std::vector<pixel*>& rows() { return my_rows; }

private:
    void reset(int w, int h);

private:
    //don't allow copying
    image(const image&);
    void operator=(const image&);

private:
    int my_width;
    int my_height;
    int my_padSize;

    std::vector<pixel> my_data; //raw raster data
    std::vector<pixel*> my_rows;

    //data structures 'file' and 'info' are using to store an image as BMP file
    //for more details see https://en.wikipedia.org/wiki/BMP_file_format
    using BITMAPFILEHEADER = struct {
        std::uint16_t sizeRest; // field is not from specification,
                            // was added for alignemt. store size of rest of the fields
        std::uint16_t type;
        std::uint32_t size;
        std::uint32_t reserved;
        std::uint32_t offBits;
    };
    BITMAPFILEHEADER file;

    using BITMAPINFOHEADER = struct {
        std::uint32_t size;
        std::int32_t width;
        std::int32_t height;
        std::uint16_t planes;
        std::uint16_t bitCount;
        std::uint32_t compression;
        std::uint32_t sizeImage;
        std::int32_t xPelsPerMeter;
        std::int32_t yPelsPerMeter;
        std::uint32_t clrUsed;
        std::uint32_t clrImportant;
    };
    BITMAPINFOHEADER info;
};
