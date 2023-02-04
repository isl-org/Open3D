// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/t/io/ImageIO.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "open3d/core/ParallelFor.h"
#include "open3d/io/ImageIO.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace t {
namespace io {

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::Image &)>>
        file_extension_to_image_read_function{
                {"png", ReadImageFromPNG},
                {"jpg", ReadImageFromJPG},
                {"jpeg", ReadImageFromJPG},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const geometry::Image &, int)>>
        file_extension_to_image_write_function{
                {"png", WriteImageToPNG},
                {"jpg", WriteImageToJPG},
                {"jpeg", WriteImageToJPG},
        };

std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename) {
    auto image = std::make_shared<geometry::Image>();
    ReadImage(filename, *image);
    return image;
}

bool ReadImage(const std::string &filename, geometry::Image &image) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::Image failed: missing file extension.");
        return false;
    }
    auto map_itr = file_extension_to_image_read_function.find(filename_ext);
    if (map_itr == file_extension_to_image_read_function.end()) {
        utility::LogWarning(
                "Read geometry::Image failed: file extension {} unknown",
                filename_ext);
        return false;
    }
    return map_itr->second(filename, image);
}

bool WriteImage(const std::string &filename,
                const geometry::Image &image,
                int quality /* = kOpen3DImageIODefaultQuality*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_image_write_function.find(filename_ext);
    if (map_itr == file_extension_to_image_write_function.end()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    return map_itr->second(filename, image.To(core::Device("CPU:0")), quality);
}

DepthNoiseSimulator::DepthNoiseSimulator(const std::string &noise_model_path) {
    // data = np.loadtxt(fname, comments='%', skiprows=5)
    const char comment_prefix = '%';
    const int skip_first_n_lines = 5;
    utility::filesystem::CFile file;
    if (!file.Open(noise_model_path, "r")) {
        utility::LogError("Read depth model failed: unable to open file: {}",
                          noise_model_path);
    }
    std::vector<float> data;
    const char *line_buffer;
    for (int i = 0; i < skip_first_n_lines; ++i) {
        if (!(line_buffer = file.ReadLine())) {
            utility::LogError(
                    "Read depth model failed: file {} is less than {} lines.",
                    noise_model_path, skip_first_n_lines);
        }
    }
    while ((line_buffer = file.ReadLine())) {
        std::string line(line_buffer);
        line.erase(std::find(line.begin(), line.end(), comment_prefix),
                   line.end());
        if (!line.empty()) {
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                data.push_back(value);
            }
        }
    }

    model_ = core::Tensor::Zeros({80, 80, 5}, core::Float32,
                                 core::Device("CPU:0"));
    geometry::kernel::TArrayIndexer<int> model_indexer(model_, 3);

    for (int y = 0; y < 80; ++y) {
        for (int x = 0; x < 80; ++x) {
            int idx = (y * 80 + x) * 23 + 3;
            bool all_less_than_8000 = true;
            for (int i = 0; i < 5; ++i) {
                if (data[idx + i] >= 8000) {
                    all_less_than_8000 = false;
                    break;
                }
            }
            if (all_less_than_8000) {
                // model_[y, x, :] = 0
                continue;
            } else {
                for (int i = 0; i < 5; ++i) {
                    *model_indexer.GetDataPtr<float>(i, x, y) =
                            data[idx + 15 + i];
                }
            }
        }
    }
}

geometry::Image DepthNoiseSimulator::Simulate(const geometry::Image &im_src,
                                              float depth_scale) {
    // Sanity checks.
    if (im_src.GetDtype() == core::Float32) {
        if (depth_scale != 1.0) {
            utility::LogWarning(
                    "Depth scale is ignored when input depth is float32.");
        }
    } else if (im_src.GetDtype() == core::UInt16) {
        if (depth_scale <= 0.0) {
            utility::LogError("Depth scale must be positive.");
        }
    } else {
        utility::LogError("Unsupported depth image dtype: {}.",
                          im_src.GetDtype().ToString());
    }
    if (im_src.GetChannels() != 1) {
        utility::LogError("Depth image must have 1 channel.");
    }

    core::Tensor im_src_tensor = im_src.AsTensor();
    const core::Device &original_device = im_src_tensor.GetDevice();
    const core::Dtype &original_dtype = im_src_tensor.GetDtype();
    int width = im_src.GetCols();
    int height = im_src.GetRows();

    im_src_tensor = im_src_tensor.To(core::Device("CPU:0")).Contiguous();
    if (original_dtype == core::UInt16) {
        im_src_tensor = im_src_tensor.To(core::Float32) / depth_scale;
    }
    core::Tensor im_dst_tensor = im_src_tensor.Clone();

    utility::random::NormalGenerator<float> gen_coord(0, 0.25);
    utility::random::NormalGenerator<float> gen_depth(0, 0.027778);

    geometry::kernel::TArrayIndexer<int> src_indexer(im_src_tensor, 2);
    geometry::kernel::TArrayIndexer<int> dst_indexer(im_dst_tensor, 2);
    geometry::kernel::TArrayIndexer<int> model_indexer(model_, 3);

    // To match the original implementation, we try to keep the same variable
    // names with reference to the original code. Compared to the original
    // implementation, parallelization is done in im_dst_tensor per-pixel level,
    // instead of per-image level. Check out the original code at:
    // http://redwood-data.org/indoor/data/simdepth.py.
    core::ParallelFor(
            core::Device("CPU:0"), width * height,
            [&] OPEN3D_DEVICE(int workload_idx) {
                // TArrayIndexer has reverted coordinate order, use (c, r).
                int r;
                int c;
                src_indexer.WorkloadToCoord(workload_idx, &c, &r);

                // Pixel shuffle.
                int x, y;
                float x_noise = deterministic_debug_mode_ ? 0 : gen_coord();
                float y_noise = deterministic_debug_mode_ ? 0 : gen_coord();
                x = std::min(std::max(static_cast<int>(round(c + x_noise)), 0),
                             width - 1);
                y = std::min(std::max(static_cast<int>(round(r + y_noise)), 0),
                             height - 1);

                // Down sample.
                float d = *src_indexer.GetDataPtr<float>(x - x % 2, y - y % 2);

                // Distortion.
                int i2 = static_cast<int>((d + 1) / 2);
                int i1 = i2 - 1;
                float a_ = (d - (i1 * 2 + 1)) / 2;
                int x_ = static_cast<int>(x / 8);
                int y_ = static_cast<int>(y / 6);
                float model_val0 = *model_indexer.GetDataPtr<float>(
                        std::min(std::max(i1, 0), 4), x_, y_);
                float model_val1 = *model_indexer.GetDataPtr<float>(
                        std::min(i2, 4), x_, y_);
                float f = (1 - a_) * model_val0 + a_ * model_val1;
                if (f == 0) {
                    d = 0;
                } else {
                    d = d / f;
                }

                // Quantization and high freq noise.
                float dst_d;
                if (d == 0) {
                    dst_d = 0;
                } else {
                    float d_noise = deterministic_debug_mode_ ? 0 : gen_depth();
                    dst_d = 35.130 * 8 / round((35.130 / d + d_noise) * 8);
                }
                *dst_indexer.GetDataPtr<float>(c, r) = dst_d;
            });

    if (original_dtype == core::UInt16) {
        im_dst_tensor = (im_dst_tensor * depth_scale).To(core::UInt16);
    }
    im_dst_tensor = im_dst_tensor.To(original_device);

    return geometry::Image(im_dst_tensor);
}

}  // namespace io
}  // namespace t
}  // namespace open3d
