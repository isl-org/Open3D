// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Open3D/Geometry/Image.h"

namespace {
/// Isotropic 2D kernels are separable:
/// two 1D kernels are applied in x and y direction.
const std::vector<double> Gaussian3 = {0.25, 0.5, 0.25};
const std::vector<double> Gaussian5 = {0.0625, 0.25, 0.375, 0.25, 0.0625};
const std::vector<double> Gaussian7 = {0.03125, 0.109375, 0.21875, 0.28125,
                                       0.21875, 0.109375, 0.03125};
const std::vector<double> Sobel31 = {-1.0, 0.0, 1.0};
const std::vector<double> Sobel32 = {1.0, 2.0, 1.0};
}  // unnamed namespace

namespace open3d {
namespace geometry {

Image &Image::Clear() {
    width_ = 0;
    height_ = 0;
    num_of_channels_ = 0;
    bytes_per_channel_ = 0;
    data_.clear();
    return *this;
}

bool Image::IsEmpty() const { return !HasData(); }

Eigen::Vector2d Image::GetMinBound() const { return Eigen::Vector2d(0.0, 0.0); }

Eigen::Vector2d Image::GetMaxBound() const {
    return Eigen::Vector2d(width_, height_);
}

bool Image::TestImageBoundary(double u,
                              double v,
                              double inner_margin /* = 0.0 */) const {
    return (u >= inner_margin && u < width_ - inner_margin &&
            v >= inner_margin && v < height_ - inner_margin);
}

std::pair<bool, double> Image::FloatValueAt(double u, double v) const {
    if ((num_of_channels_ != 1) || (bytes_per_channel_ != 4) ||
        (u < 0.0 || u > (double)(width_ - 1) || v < 0.0 ||
         v > (double)(height_ - 1))) {
        return std::make_pair(false, 0.0);
    }
    int ui = std::max(std::min((int)u, width_ - 2), 0);
    int vi = std::max(std::min((int)v, height_ - 2), 0);
    double pu = u - ui;
    double pv = v - vi;
    float value[4] = {*PointerAt<float>(ui, vi), *PointerAt<float>(ui, vi + 1),
                      *PointerAt<float>(ui + 1, vi),
                      *PointerAt<float>(ui + 1, vi + 1)};
    return std::make_pair(true,
                          (value[0] * (1 - pv) + value[1] * pv) * (1 - pu) +
                                  (value[2] * (1 - pv) + value[3] * pv) * pu);
}

template <typename T>
T *Image::PointerAt(int u, int v) const {
    return (T *)(data_.data() + (v * width_ + u) * sizeof(T));
}

template float *Image::PointerAt<float>(int u, int v) const;
template int *Image::PointerAt<int>(int u, int v) const;
template uint8_t *Image::PointerAt<uint8_t>(int u, int v) const;
template uint16_t *Image::PointerAt<uint16_t>(int u, int v) const;

template <typename T>
T *Image::PointerAt(int u, int v, int ch) const {
    return (T *)(data_.data() +
                 ((v * width_ + u) * num_of_channels_ + ch) * sizeof(T));
}

template float *Image::PointerAt<float>(int u, int v, int ch) const;
template int *Image::PointerAt<int>(int u, int v, int ch) const;
template uint8_t *Image::PointerAt<uint8_t>(int u, int v, int ch) const;
template uint16_t *Image::PointerAt<uint16_t>(int u, int v, int ch) const;

std::shared_ptr<Image> Image::ConvertDepthToFloatImage(
        double depth_scale /* = 1000.0*/, double depth_trunc /* = 3.0*/) const {
    // don't need warning message about image type
    // as we call CreateFloatImage
    auto output = CreateFloatImage();
    for (int y = 0; y < output->height_; y++) {
        for (int x = 0; x < output->width_; x++) {
            float *p = output->PointerAt<float>(x, y);
            *p /= (float)depth_scale;
            if (*p >= depth_trunc) *p = 0.0f;
        }
    }
    return output;
}

Image &Image::ClipIntensity(double min /* = 0.0*/, double max /* = 1.0*/) {
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[ClipIntensity] Unsupported image format.");
    }
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            float *p = PointerAt<float>(x, y);
            if (*p > max) *p = (float)max;
            if (*p < min) *p = (float)min;
        }
    }
    return *this;
}

Image &Image::LinearTransform(double scale, double offset /* = 0.0*/) {
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[LinearTransform] Unsupported image format.");
    }
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            float *p = PointerAt<float>(x, y);
            (*p) = (float)(scale * (*p) + offset);
        }
    }
    return *this;
}

std::shared_ptr<Image> Image::Downsample() const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[Downsample] Unsupported image format.");
    }
    int half_width = (int)floor((double)width_ / 2.0);
    int half_height = (int)floor((double)height_ / 2.0);
    output->Prepare(half_width, half_height, 1, 4);

#ifdef _OPENMP
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for collapse(2) schedule(static)
#endif
#endif
    for (int y = 0; y < output->height_; y++) {
        for (int x = 0; x < output->width_; x++) {
            float *p1 = PointerAt<float>(x * 2, y * 2);
            float *p2 = PointerAt<float>(x * 2 + 1, y * 2);
            float *p3 = PointerAt<float>(x * 2, y * 2 + 1);
            float *p4 = PointerAt<float>(x * 2 + 1, y * 2 + 1);
            float *p = output->PointerAt<float>(x, y);
            *p = (*p1 + *p2 + *p3 + *p4) / 4.0f;
        }
    }
    return output;
}

std::shared_ptr<Image> Image::FilterHorizontal(
        const std::vector<double> &kernel) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4 ||
        kernel.size() % 2 != 1) {
        utility::LogError(
                "[FilterHorizontal] Unsupported image format or kernel "
                "size.");
    }
    output->Prepare(width_, height_, 1, 4);

    const int half_kernel_size = (int)(floor((double)kernel.size() / 2.0));

#ifdef _OPENMP
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for collapse(2) schedule(static)
#endif
#endif
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            float *po = output->PointerAt<float>(x, y, 0);
            double temp = 0;
            for (int i = -half_kernel_size; i <= half_kernel_size; i++) {
                int x_shift = x + i;
                if (x_shift < 0) x_shift = 0;
                if (x_shift > width_ - 1) x_shift = width_ - 1;
                float *pi = PointerAt<float>(x_shift, y, 0);
                temp += (*pi * (float)kernel[i + half_kernel_size]);
            }
            *po = (float)temp;
        }
    }
    return output;
}

std::shared_ptr<Image> Image::Filter(Image::FilterType type) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[Filter] Unsupported image format.");
    }

    switch (type) {
        case Image::FilterType::Gaussian3:
            output = Filter(Gaussian3, Gaussian3);
            break;
        case Image::FilterType::Gaussian5:
            output = Filter(Gaussian5, Gaussian5);
            break;
        case Image::FilterType::Gaussian7:
            output = Filter(Gaussian7, Gaussian7);
            break;
        case Image::FilterType::Sobel3Dx:
            output = Filter(Sobel31, Sobel32);
            break;
        case Image::FilterType::Sobel3Dy:
            output = Filter(Sobel32, Sobel31);
            break;
        default:
            utility::LogError("[Filter] Unsupported filter type.");
            break;
    }
    return output;
}

ImagePyramid Image::FilterPyramid(const ImagePyramid &input,
                                  Image::FilterType type) {
    std::vector<std::shared_ptr<Image>> output;
    for (size_t i = 0; i < input.size(); i++) {
        auto layer_filtered = input[i]->Filter(type);
        output.push_back(layer_filtered);
    }
    return output;
}

std::shared_ptr<Image> Image::Filter(const std::vector<double> &dx,
                                     const std::vector<double> &dy) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[Filter] Unsupported image format.");
    }

    auto temp1 = FilterHorizontal(dx);
    auto temp2 = temp1->Transpose();
    auto temp3 = temp2->FilterHorizontal(dy);
    auto temp4 = temp3->Transpose();
    return temp4;
}

std::shared_ptr<Image> Image::Transpose() const {
    auto output = std::make_shared<Image>();
    output->Prepare(height_, width_, num_of_channels_, bytes_per_channel_);

    int out_bytes_per_line = output->BytesPerLine();
    int in_bytes_per_line = BytesPerLine();
    int bytes_per_pixel = num_of_channels_ * bytes_per_channel_;

#ifdef _OPENMP
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for collapse(2) schedule(static)
#endif
#endif
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            std::copy(
                    data_.data() + y * in_bytes_per_line + x * bytes_per_pixel,
                    data_.data() + y * in_bytes_per_line +
                            (x + 1) * bytes_per_pixel,
                    output->data_.data() + x * out_bytes_per_line +
                            y * bytes_per_pixel);
        }
    }

    return output;
}

std::shared_ptr<Image> Image::FlipVertical() const {
    auto output = std::make_shared<Image>();
    output->Prepare(width_, height_, num_of_channels_, bytes_per_channel_);

    int bytes_per_line = BytesPerLine();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height_; y++) {
        std::copy(data_.data() + y * bytes_per_line,
                  data_.data() + (y + 1) * bytes_per_line,
                  output->data_.data() + (height_ - y - 1) * bytes_per_line);
    }
    return output;
}

std::shared_ptr<Image> Image::FlipHorizontal() const {
    auto output = std::make_shared<Image>();
    output->Prepare(width_, height_, num_of_channels_, bytes_per_channel_);

    int bytes_per_line = BytesPerLine();
    int bytes_per_pixel = num_of_channels_ * bytes_per_channel_;
#ifdef _OPENMP
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for collapse(2) schedule(static)
#endif
#endif
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            std::copy(data_.data() + y * bytes_per_line + x * bytes_per_pixel,
                      data_.data() + y * bytes_per_line +
                              (x + 1) * bytes_per_pixel,
                      output->data_.data() + y * bytes_per_line +
                              (width_ - x - 1) * bytes_per_pixel);
        }
    }

    return output;
}

std::shared_ptr<Image> Image::Dilate(int half_kernel_size /* = 1 */) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 1) {
        utility::LogError("[Dilate] Unsupported image format.");
    }
    output->Prepare(width_, height_, 1, 1);

#ifdef _OPENMP
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for collapse(2) schedule(static)
#endif
#endif
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            for (int yy = -half_kernel_size; yy <= half_kernel_size; yy++) {
                for (int xx = -half_kernel_size; xx <= half_kernel_size; xx++) {
                    unsigned char *pi;
                    if (TestImageBoundary(x + xx, y + yy)) {
                        pi = PointerAt<unsigned char>(x + xx, y + yy);
                        if (*pi == 255) {
                            *output->PointerAt<unsigned char>(x, y, 0) = 255;
                            xx = half_kernel_size;
                            yy = half_kernel_size;
                        }
                    }
                }
            }
        }
    }
    return output;
}

std::shared_ptr<Image> Image::CreateDepthBoundaryMask(
        double depth_threshold_for_discontinuity_check,
        int half_dilation_kernel_size_for_discontinuity_map) const {
    auto depth_image = CreateFloatImage();  // necessary?
    int width = depth_image->width_;
    int height = depth_image->height_;
    auto depth_image_gradient_dx =
            depth_image->Filter(Image::FilterType::Sobel3Dx);
    auto depth_image_gradient_dy =
            depth_image->Filter(Image::FilterType::Sobel3Dy);
    auto mask = std::make_shared<Image>();
    mask->Prepare(width, height, 1, 1);

#ifdef _OPENMP
#ifdef _WIN32
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for collapse(2) schedule(static)
#endif
#endif
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            double dx = *depth_image_gradient_dx->PointerAt<float>(u, v);
            double dy = *depth_image_gradient_dy->PointerAt<float>(u, v);
            double mag = sqrt(dx * dx + dy * dy);
            if (mag > depth_threshold_for_discontinuity_check) {
                *mask->PointerAt<unsigned char>(u, v) = 255;
            } else {
                *mask->PointerAt<unsigned char>(u, v) = 0;
            }
        }
    }
    if (half_dilation_kernel_size_for_discontinuity_map >= 1) {
        auto mask_dilated =
                mask->Dilate(half_dilation_kernel_size_for_discontinuity_map);
        return mask_dilated;
    } else {
        return mask;
    }
}

}  // namespace geometry
}  // namespace open3d
