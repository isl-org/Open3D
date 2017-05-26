// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "Helper.h"

namespace {

	// some parameters
	#define minDepth		0.f			//in meters (0.0)
	#define maxDepth		4.f			//in meters (4.0)	

} // unnamed namespace


namespace three{


// this function comes from helper.cpp in IntegrateRGBD
void ConvertDepthToFloatImage(const Image &depth, Image &depth_f,
  double depth_scale/* = 1000.0*/, double depth_trunc/* = 3.0*/)
{
  if (depth_f.IsEmpty()) {
    depth_f.PrepareImage(depth.width_, depth.height_, 1, 4);
  }
  float *p = (float *)depth_f.data_.data();
  const uint16_t *pi = (const uint16_t *)depth.data_.data();
  for (int i = 0; i < depth.height_ * depth.width_; i++, p++, pi++) {
    *p = (float)(*pi) / (float)depth_scale;
    if (*p >= depth_trunc) {
      *p = 0.0f;
    }
  }
}

void PreprocessDepth(const Image &depth)
{
  float *p = (float *)depth.data_.data();
  for (int i = 0; i < depth.height_ * depth.width_; i++, p++) {
    if ((*p > maxDepth || *p < minDepth || *p <= 0)) {
      *p = std::numeric_limits<float>::quiet_NaN();
    }
  }
}

// 3x3 filtering
// assumes single channel float type image
std::shared_ptr<Image> FilteringImage(const Image &input, const float *kernel)
{
  auto output = std::make_shared<Image>();
  if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
    PrintDebug("[CreatePointCloudFromDepthImage] Unsupported image format.\n");
    return output;
  }
  output->PrepareImage(input.width_, input.height_, 1, 4);

  for (int y = 0; y < input.height_; y++) {
    for (int x = 0; x < input.width_; x++) {
      double sum = 0.0f;
      for (int yb = 0; yb < 3; yb++) {
        int yy = y + (yb - 1);
        if (yy < 0 | yy >= input.height_)
          continue;
        for (int xb = 0; xb < 3; xb++) {
          int xx = x + (xb - 1);
          if (xx < 0 | xx >= input.width_)
            continue;
          // do we have user intuitive way to access pixel value?
          float *pi = (float *)input.data_.data() +
            (y * input.width_ + x);
          sum += *pi * kernel[yb * 3 + xb];
        }
      }
      float *po = (float *)output->data_.data() +
        (y * input.width_ + x);
      *po = sum;
    }
  }
}

// 2x image downsampling
// assumes float type image
// simple 2x2 averaging
// assumes 2x powered image width and height
// need to double check how we are going to handle invalid depth
std::shared_ptr<Image> DownsamplingImage(const Image &input)
{
  auto output = std::make_shared<Image>();
  if (input.num_of_channels_ != 1 ||
    input.bytes_per_channel_ != 4 ||
    (input.width_ % 2 != 0 || input.height_ % 2 != 0)) {
    PrintDebug("[CreatePointCloudFromDepthImage] Unsupported image format.\n");
    return output;
  }
  output->PrepareImage(input.width_/2, input.height_/2, 1, 4);

  float* inputdata = (float *)input.data_.data();
  float* outputdata = (float *)output->data_.data();
  for (int c = 0; c < output->num_of_channels_; c++) {
    int cpad = c * output->width_ * output->height_;
    for (int y = 0; y < output->height_; y++) {
      for (int x = 0; x < output->width_; x++) {
        float *p1 = inputdata + (cpad + y * 2 * output->width_ + x * 2);
        float *p2 = inputdata + (cpad + y * 2 * output->width_ + (x + 1) * 2);
        float *p3 = inputdata + (cpad + (y + 1) * 2 * output->width_ + x * 2);
        float *p4 = inputdata + (cpad + (y + 1) * 2 * output->width_ + (x + 1) * 2);
        float *p = outputdata + (cpad + y * output->width_ + x);
        *p = (*p1 + *p2 + *p3 + *p4) / 4.0f;
      }
    }
  }
}

void BuildingPyramidImage(const Image& image,
  std::vector<std::shared_ptr<const Image>>& pyramidImage,
  size_t levelCount)
{
  // build image pyramid
  pyramidImage.clear(); // is this good for clearing? it might have some existing data

  for (int i = 0; i < levelCount; i++) {
    if (i == 0) {
      // no downsampling, make copy
      // need to check preferred way to hardcopy
      std::shared_ptr<Image> image_copy_ptr(new Image);
      *image_copy_ptr = image;
      pyramidImage.push_back(image_copy_ptr);
    }
    else {
      // image blur and downsampling
      auto layer_b = FilteringImage(*pyramidImage[i - 1], Gaussian);
      auto layer_bd = DownsamplingImage(*layer_b);
      pyramidImage.push_back(layer_bd);
    }
  }
}

void BuildingPyramidImage(
  const Image &image,
  std::vector<std::shared_ptr<const Image>> &pyramidImage,
  std::vector<std::shared_ptr<const Image>> &pyramidImageGradx,
  std::vector<std::shared_ptr<const Image>> &pyramidImageGrady,
  size_t levelCount)
{
  BuildingPyramidImage(image, pyramidImage, levelCount);

  // make image pyramid by applying sobel filter
  pyramidImageGradx.clear();
  pyramidImageGrady.clear();
  for (int i = 0; i < levelCount; i++) {
    if (i == 0) {
      auto image_dx = FilteringImage(image, Sobel_dx);
      auto image_dy = FilteringImage(image, Sobel_dy);
      pyramidImageGradx.push_back(image_dx);
      pyramidImageGrady.push_back(image_dy);
    }
    else {
      auto image_dx = FilteringImage(*pyramidImage[i - 1], Sobel_dx);
      auto image_dy = FilteringImage(*pyramidImage[i - 1], Sobel_dy);
      pyramidImageGradx.push_back(image_dx);
      pyramidImageGrady.push_back(image_dy);
    }
  }
}

}	// namespace three
