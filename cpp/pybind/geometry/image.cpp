// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/Image.h"

#include "open3d/geometry/RGBDImage.h"
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

// Image functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"color", "The color image."},
                {"convert_rgb_to_intensity",
                 "Whether to convert RGB image to intensity image."},
                {"depth", "The depth image."},
                {"depth_scale",
                 "The ratio to scale depth values. The depth values will first "
                 "be scaled and then truncated."},
                {"depth_trunc",
                 "Depth values larger than ``depth_trunc`` gets truncated to "
                 "0. The depth values will first be scaled and then "
                 "truncated."},
                {"filter_type", "The filter type to be applied."},
                {"image", "The Image object."},
                {"image_pyramid", "The ImagePyramid object"},
                {"num_of_levels ", "Levels of the image pyramid"},
                {"with_gaussian_filter",
                 "When ``True``, image in the pyramid will first be filtered "
                 "by a 3x3 Gaussian kernel before downsampling."}};

void pybind_image_declarations(py::module &m) {
    py::enum_<Image::FilterType> image_filter_type(m, "ImageFilterType");
    image_filter_type.value("Gaussian3", Image::FilterType::Gaussian3)
            .value("Gaussian5", Image::FilterType::Gaussian5)
            .value("Gaussian7", Image::FilterType::Gaussian7)
            .value("Sobel3dx", Image::FilterType::Sobel3Dx)
            .value("Sobel3dy", Image::FilterType::Sobel3Dy)
            .export_values();
    image_filter_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Image filter types.";
            }),
            py::none(), py::none(), "");
    py::class_<Image, PyGeometry2D<Image>, std::shared_ptr<Image>, Geometry2D>
            image(m, "Image", py::buffer_protocol(),
                  "The image class stores image with customizable width, "
                  "height, num of channels and bytes per channel.");
    py::class_<RGBDImage, PyGeometry2D<RGBDImage>, std::shared_ptr<RGBDImage>,
               Geometry2D>
            rgbd_image(m, "RGBDImage",
                       "RGBDImage is for a pair of registered color and depth "
                       "images, viewed from the same view, of the same "
                       "resolution. If you have other format, convert it "
                       "first.");
}
void pybind_image_definitions(py::module &m) {
    auto image = static_cast<py::class_<Image, PyGeometry2D<Image>,
                                        std::shared_ptr<Image>, Geometry2D>>(
            m.attr("Image"));
    py::detail::bind_default_constructor<Image>(image);
    py::detail::bind_copy_functions<Image>(image);
    image.def(py::init([](py::buffer b) {
             py::buffer_info info = b.request();
             int width, height, num_of_channels = 0, bytes_per_channel;
             if (info.format == py::format_descriptor<uint8_t>::format() ||
                 info.format == py::format_descriptor<int8_t>::format()) {
                 bytes_per_channel = 1;
             } else if (info.format ==
                                py::format_descriptor<uint16_t>::format() ||
                        info.format ==
                                py::format_descriptor<int16_t>::format()) {
                 bytes_per_channel = 2;
             } else if (info.format == py::format_descriptor<float>::format()) {
                 bytes_per_channel = 4;
             } else {
                 throw std::runtime_error(
                         "Image can only be initialized from buffer of uint8, "
                         "uint16, or float!");
             }
             if (info.strides[info.ndim - 1] != bytes_per_channel) {
                 throw std::runtime_error(
                         "Image can only be initialized from c-style buffer.");
             }
             if (info.ndim == 2) {
                 num_of_channels = 1;
             } else if (info.ndim == 3) {
                 num_of_channels = (int)info.shape[2];
             }
             height = (int)info.shape[0];
             width = (int)info.shape[1];
             if (info.strides[1] != num_of_channels * bytes_per_channel ||
                 info.strides[0] !=
                         width * num_of_channels * bytes_per_channel) {
                 throw std::runtime_error(
                         "Image can only be initialized from a contiguous "
                         "buffer.");
             }
             auto img = new Image();
             img->Prepare(width, height, num_of_channels, bytes_per_channel);
             memcpy(img->data_.data(), info.ptr, img->data_.size());
             return img;
         }))
            .def_buffer([](Image &img) -> py::buffer_info {
                std::string format;
                switch (img.bytes_per_channel_) {
                    case 1:
                        format = py::format_descriptor<uint8_t>::format();
                        break;
                    case 2:
                        format = py::format_descriptor<uint16_t>::format();
                        break;
                    case 4:
                        format = py::format_descriptor<float>::format();
                        break;
                    default:
                        throw std::runtime_error(
                                "Image has unrecognized bytes_per_channel.");
                        break;
                }
                if (img.num_of_channels_ == 1) {
                    return py::buffer_info(
                            img.data_.data(), img.bytes_per_channel_, format, 2,
                            {static_cast<unsigned long>(img.height_),
                             static_cast<unsigned long>(img.width_)},
                            {static_cast<unsigned long>(img.bytes_per_channel_ *
                                                        img.num_of_channels_ *
                                                        img.width_),
                             static_cast<unsigned long>(img.bytes_per_channel_ *
                                                        img.num_of_channels_)});
                } else {
                    return py::buffer_info(
                            img.data_.data(), img.bytes_per_channel_, format, 3,
                            {static_cast<unsigned long>(img.height_),
                             static_cast<unsigned long>(img.width_),
                             static_cast<unsigned long>(img.num_of_channels_)},
                            {static_cast<unsigned long>(img.bytes_per_channel_ *
                                                        img.num_of_channels_ *
                                                        img.width_),
                             static_cast<unsigned long>(img.bytes_per_channel_ *
                                                        img.num_of_channels_),
                             static_cast<unsigned long>(
                                     img.bytes_per_channel_)});
                }
            })
            .def("__repr__",
                 [](const Image &img) {
                     return std::string("Image of size ") +
                            std::to_string(img.width_) + std::string("x") +
                            std::to_string(img.height_) + ", with " +
                            std::to_string(img.num_of_channels_) +
                            std::string(
                                    " channels.\nUse numpy.asarray to access "
                                    "buffer "
                                    "data.");
                 })
            .def(
                    "filter",
                    [](const Image &input, Image::FilterType filter_type) {
                        if (input.num_of_channels_ != 1 ||
                            input.bytes_per_channel_ != 4) {
                            auto input_f = input.CreateFloatImage();
                            auto output = input_f->Filter(filter_type);
                            return *output;
                        } else {
                            auto output = input.Filter(filter_type);
                            return *output;
                        }
                    },
                    "Function to filter Image", "filter_type"_a)
            .def("flip_vertical", &Image::FlipVertical,
                 "Function to flip image vertically (upside down)")
            .def("flip_horizontal", &Image::FlipHorizontal,
                 "Function to flip image horizontally (from left to right)")
            .def(
                    "create_pyramid",
                    [](const Image &input, size_t num_of_levels,
                       bool with_gaussian_filter) {
                        if (input.num_of_channels_ != 1 ||
                            input.bytes_per_channel_ != 4) {
                            auto input_f = input.CreateFloatImage();
                            auto output = input_f->CreatePyramid(
                                    num_of_levels, with_gaussian_filter);
                            return output;
                        } else {
                            auto output = input.CreatePyramid(
                                    num_of_levels, with_gaussian_filter);
                            return output;
                        }
                    },
                    "Function to create ImagePyramid", "num_of_levels"_a,
                    "with_gaussian_filter"_a)
            .def_static(
                    "filter_pyramid",
                    [](const ImagePyramid &input,
                       Image::FilterType filter_type) {
                        auto output = Image::FilterPyramid(input, filter_type);
                        return output;
                    },
                    "Function to filter ImagePyramid", "image_pyramid"_a,
                    "filter_type"_a);

    docstring::ClassMethodDocInject(m, "Image", "filter",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "Image", "create_pyramid",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "Image", "filter_pyramid",
                                    map_shared_argument_docstrings);

    auto rgbd_image =
            static_cast<py::class_<RGBDImage, PyGeometry2D<RGBDImage>,
                                   std::shared_ptr<RGBDImage>, Geometry2D>>(
                    m.attr("RGBDImage"));
    py::detail::bind_default_constructor<RGBDImage>(rgbd_image);
    rgbd_image
            .def_readwrite("color", &RGBDImage::color_,
                           "open3d.geometry.Image: The color image.")
            .def_readwrite("depth", &RGBDImage::depth_,
                           "open3d.geometry.Image: The depth image.")
            .def("__repr__",
                 [](const RGBDImage &rgbd_image) {
                     return std::string("RGBDImage of size \n") +
                            std::string("Color image : ") +
                            std::to_string(rgbd_image.color_.width_) +
                            std::string("x") +
                            std::to_string(rgbd_image.color_.height_) +
                            ", with " +
                            std::to_string(rgbd_image.color_.num_of_channels_) +
                            std::string(" channels.\n") +
                            std::string("Depth image : ") +
                            std::to_string(rgbd_image.depth_.width_) +
                            std::string("x") +
                            std::to_string(rgbd_image.depth_.height_) +
                            ", with " +
                            std::to_string(rgbd_image.depth_.num_of_channels_) +
                            std::string(" channels.\n") +
                            std::string(
                                    "Use numpy.asarray to access buffer data.");
                 })
            .def_static("create_from_color_and_depth",
                        &RGBDImage::CreateFromColorAndDepth,
                        "Function to make RGBDImage from color and depth image",
                        "color"_a, "depth"_a, "depth_scale"_a = 1000.0,
                        "depth_trunc"_a = 3.0,
                        "convert_rgb_to_intensity"_a = true)
            .def_static("create_from_redwood_format",
                        &RGBDImage::CreateFromRedwoodFormat,
                        "Function to make RGBDImage (for Redwood format)",
                        "color"_a, "depth"_a,
                        "convert_rgb_to_intensity"_a = true)
            .def_static(
                    "create_from_tum_format", &RGBDImage::CreateFromTUMFormat,
                    "Function to make RGBDImage (for TUM format)", "color"_a,
                    "depth"_a, "convert_rgb_to_intensity"_a = true)
            .def_static(
                    "create_from_sun_format", &RGBDImage::CreateFromSUNFormat,
                    "Function to make RGBDImage (for SUN format)", "color"_a,
                    "depth"_a, "convert_rgb_to_intensity"_a = true)
            .def_static(
                    "create_from_nyu_format", &RGBDImage::CreateFromNYUFormat,
                    "Function to make RGBDImage (for NYU format)", "color"_a,
                    "depth"_a, "convert_rgb_to_intensity"_a = true);

    docstring::ClassMethodDocInject(m, "RGBDImage",
                                    "create_from_color_and_depth",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RGBDImage",
                                    "create_from_redwood_format",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RGBDImage", "create_from_tum_format",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RGBDImage", "create_from_sun_format",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RGBDImage", "create_from_nyu_format",
                                    map_shared_argument_docstrings);
}

}  // namespace geometry
}  // namespace open3d
