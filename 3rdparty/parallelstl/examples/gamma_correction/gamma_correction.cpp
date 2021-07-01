// -*- C++ -*-
//===-- gamma_correction.cpp ----------------------------------------------===//
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

#include <iostream>
#include <cmath>
#include <cassert>

#include "pstl/algorithm"
#include "pstl/execution"
#include "utils.h"

//! fractal class
class fractal {
public:
    //! Constructor
    fractal(int x, int y): my_size{x, y} {}
    //! One pixel calculation routine
    double calcOnePixel(int x, int y);

private:
    //! Size of the fractal area
    const int my_size[2];
    //! Fractal properties
    double cx = -0.7436;
    const double cy = 0.1319;
    const double magn = 2000000.0;
    const int max_iter = 1000;
};

double fractal::calcOnePixel(int x0, int y0) {
    double fx0 = double(x0) - double(my_size[0]) / 2;
    double fy0 = double(y0) - double(my_size[1]) / 2;
    fx0 = fx0 / magn + cx;
    fy0 = fy0 / magn + cy;

    double res = 0, x = 0, y = 0;
    for(int iter = 0; x*x + y*y <= 4 && iter < max_iter; ++iter) {
        const double val = x*x - y*y + fx0;
        y = 2*x*y + fy0, x = val;
        res += exp(-sqrt(x*x+y*y));
    }

    return res;
}

template<typename Rows>
void applyGamma(Rows& image, double g) {
    typedef decltype(image[0]) Row;
    typedef decltype(image[0][0]) Pixel;
    const int w = image[1] - image[0];

    //execution STL algorithms with execution policies - pstl::execution::par and pstl::execution::unseq
    std::for_each(pstl::execution::par, image.begin(), image.end(), [g, w](Row& r) {
        std::transform(pstl::execution::unseq, r, r+w, r, [g](Pixel& p) {
            double v = 0.3*p.bgra[2] + 0.59*p.bgra[1] + 0.11*p.bgra[0]; //RGB Luminance value
            assert(v > 0);
            auto res = static_cast<std::uint8_t>(255.0 * pow(v / 255.0, g));
            return image::pixel(res, res, res);
        });
    });
}

int main(int argc, char* argv[]) {

    //create a fractal image
    image img(800, 800);
    fractal fr(img.width(), img.height());
    img.fill([&fr](int x, int y) { return fr.calcOnePixel(x, y); });
    img.write("image_1.bmp");

    //apply gamma
    applyGamma(img.rows(), 1.5);

    //write result to disk
    img.write("image_1_gamma.bmp");
    std::cout<<"done"<<std::endl;

    return 0;
}
