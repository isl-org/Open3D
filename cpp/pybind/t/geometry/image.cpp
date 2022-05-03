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

#include "open3d/t/geometry/Image.h"

#include <string>
#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "pybind/docstring.h"
#include "pybind/pybind_utils.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

// Image functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"color", "The color image."},
                {"depth", "The depth image."},
                {"aligned",
                 "Are the two images aligned (same viewpoint and resolution)?"},
                {"image", "The Image object."},
                {"tensor",
                 "Tensor of the image. The tensor must be contiguous. The "
                 "tensor must be 2D (rows, cols) or 3D (rows, cols, "
                 "channels)."},
                {"rows",
                 "Number of rows of the image, i.e. image height. rows must be "
                 "non-negative."},
                {"cols",
                 "Number of columns of the image, i.e. image width. cols must "
                 "be non-negative."},
                {"channels",
                 "Number of channels of the image. E.g. for RGB image, "
                 "channels == 3; for grayscale image, channels == 1. channels "
                 "must be greater than 0."},
                {"dtype", "Data type of the image."},
                {"device", "Device where the image is stored."},
                {"scale",
                 "First multiply image pixel values with this factor. "
                 "This should be positive for unsigned dtypes."},
                {"offset", "Then add this factor to all image pixel values."},
                {"kernel_size", "Kernel size for filters and dilations."},
                {"value_sigma", "Standard deviation for the image content."},
                {"distance_sigma",
                 "Standard deviation for the image pixel positions."}};

void pybind_image(py::module &m) {
    py::class_<Image, PyGeometry<Image>, std::shared_ptr<Image>, Geometry>
            image(m, "Image", py::buffer_protocol(),
                  "The Image class stores image with customizable rols, cols, "
                  "channels, dtype and device.");

    py::enum_<Image::InterpType>(m, "InterpType", "Interpolation type.")
            .value("Nearest", Image::InterpType::Nearest)
            .value("Linear", Image::InterpType::Linear)
            .value("Cubic", Image::InterpType::Cubic)
            .value("Lanczos", Image::InterpType::Lanczos)
            .value("Super", Image::InterpType::Super)
            .export_values();

    // Constructors
    image.def(py::init<int64_t, int64_t, int64_t, core::Dtype, core::Device>(),
              "Row-major storage is used, similar to OpenCV. Use (row, col, "
              "channel) indexing order for image creation and accessing. In "
              "general, (r, c, ch) are the preferred variable names for "
              "consistency, and avoid using width, height, u, v, x, y for "
              "coordinates.",
              "rows"_a = 0, "cols"_a = 0, "channels"_a = 1,
              "dtype"_a = core::Float32, "device"_a = core::Device("CPU:0"))
            .def(py::init<core::Tensor &>(),
                 "Construct from a tensor. The tensor won't be copied and "
                 "memory will be shared.",
                 "tensor"_a);
    docstring::ClassMethodDocInject(m, "Image", "__init__",
                                    map_shared_argument_docstrings);
    // Buffer protocol.
    image.def_buffer([](Image &I) -> py::buffer_info {
        if (I.GetDevice().GetType() != core::Device::DeviceType::CPU) {
            utility::LogError(
                    "Cannot convert image buffer since it's not on CPU. "
                    "Convert to CPU image by calling .cpu() first.");
        }
        core::SizeVector strides_in_bytes = I.AsTensor().GetStrides();
        const int64_t element_byte_size = I.GetDtype().ByteSize();
        for (size_t i = 0; i < strides_in_bytes.size(); i++) {
            strides_in_bytes[i] *= element_byte_size;
        }
        return py::buffer_info(I.GetDataPtr(), element_byte_size,
                               pybind_utils::DtypeToArrayFormat(I.GetDtype()),
                               I.AsTensor().NumDims(), I.AsTensor().GetShape(),
                               strides_in_bytes);
    });
    // Info.
    image.def_property_readonly("dtype", &Image::GetDtype,
                                "Get dtype of the image")
            .def_property_readonly("device", &Image::GetDevice,
                                   "Get the device of the image.")
            .def_property_readonly("rows", &Image::GetRows,
                                   "Get the number of rows of the image.")
            .def_property_readonly("columns", &Image::GetCols,
                                   "Get the number of columns of the image.")
            .def_property_readonly("channels", &Image::GetChannels,
                                   "Get the number of channels of the image.")
            // functions
            .def("clear", &Image::Clear, "Clear stored data.")
            .def("is_empty", &Image::IsEmpty, "Is any data stored?")
            .def("get_min_bound", &Image::GetMinBound,
                 "Compute min 2D coordinates for the data (always {0, 0}).")
            .def("get_max_bound", &Image::GetMaxBound,
                 "Compute max 2D coordinates for the data ({rows, cols}).")
            .def("linear_transform", &Image::LinearTransform,
                 "Function to linearly transform pixel intensities in place: "
                 "image = scale * image + offset.",
                 "scale"_a = 1.0, "offset"_a = 0.0)
            .def("dilate", &Image::Dilate,
                 "Return a new image after performing morphological dilation. "
                 "Supported datatypes are UInt8, UInt16 and Float32 with "
                 "{1, 3, 4} channels. An 8-connected neighborhood is used to "
                 "create the dilation mask.",
                 "kernel_size"_a = 3)
            .def("filter", &Image::Filter,
                 "Return a new image after filtering with the given kernel.",
                 "kernel"_a)
            .def("filter_gaussian", &Image::FilterGaussian,
                 "Return a new image after Gaussian filtering. "
                 "Possible kernel_size: odd numbers >= 3 are supported.",
                 "kernel_size"_a = 3, "sigma"_a = 1.0)
            .def("filter_bilateral", &Image::FilterBilateral,
                 "Return a new image after bilateral filtering."
                 "Note: CPU (IPP) and CUDA (NPP) versions are inconsistent: "
                 "CPU uses a round kernel (radius = floor(kernel_size / 2)), "
                 "while CUDA uses a square kernel (width = kernel_size). "
                 "Make sure to tune parameters accordingly.",
                 "kernel_size"_a = 3, "value_sigma"_a = 20.0,
                 "dist_sigma"_a = 10.0)
            .def("filter_sobel", &Image::FilterSobel,
                 "Return a pair of new gradient images (dx, dy) after Sobel "
                 "filtering. Possible kernel_size: 3 and 5.",
                 "kernel_size"_a = 3)
            .def("resize", &Image::Resize,
                 "Return a new image after resizing with specified "
                 "interpolation type. Downsample if sampling rate is < 1. "
                 "Upsample if sampling rate > 1. Aspect ratio is always "
                 "kept.",
                 "sampling_rate"_a = 0.5,
                 "interp_type"_a = Image::InterpType::Nearest)
            .def("pyrdown", &Image::PyrDown,
                 "Return a new downsampled image with pyramid downsampling "
                 "formed by a chained Gaussian filter (kernel_size = 5, sigma"
                 " = 1.0) and a resize (ratio = 0.5) operation.")
            .def("rgb_to_gray", &Image::RGBToGray,
                 "Converts a 3-channel RGB image to a new 1-channel Grayscale "
                 "image by I = 0.299 * R + 0.587 * G + 0.114 * B.")
            .def("__repr__", &Image::ToString);
    docstring::ClassMethodDocInject(m, "Image", "linear_transform",
                                    map_shared_argument_docstrings);

    // Depth utilities.
    image.def("clip_transform", &Image::ClipTransform,
              "Preprocess a image of shape (rows, cols, channels=1), typically"
              " used for a depth image. UInt16 and Float32 Dtypes supported. "
              "Each pixel will be transformed by\n"
              "x = x / scale\n"
              "x = x < min_value ? clip_fill : x\n"
              "x = x > max_value ? clip_fill : x\n"
              "Use INF, NAN or 0.0 (default) for clip_fill",
              "scale"_a, "min_value"_a, "max_value"_a, "clip_fill"_a = 0.0f);
    image.def("create_vertex_map", &Image::CreateVertexMap,
              "Create a vertex map of shape (rows, cols, channels=3) in Float32"
              " from an image of shape (rows, cols, channels=1) in Float32 "
              "using unprojection. The input depth is expected to be the output"
              " of clip_transform.",
              "intrinsics"_a, "invalid_fill"_a = 0.0f);
    image.def("create_normal_map", &Image::CreateNormalMap,
              "Create a normal map of shape (rows, cols, channels=3) in Float32"
              " from a vertex map of shape (rows, cols, channels=1) in Float32 "
              "using cross product of V(r, c+1)-V(r, c) and V(r+1, c)-V(r, c)"
              ". The input vertex map is expected to be the output of "
              "create_vertex_map. You may need to start with a filtered depth "
              " image (e.g. with filter_bilateral) to obtain good results.",
              "invalid_fill"_a = 0.0f);
    image.def(
            "colorize_depth", &Image::ColorizeDepth,
            "Colorize an input depth image (with Dtype UInt16 or Float32). The"
            " image values are divided by scale, then clamped within "
            "(min_value, max_value) and finally converted to a 3 channel UInt8"
            " RGB image using the Turbo colormap as a lookup table.",
            "scale"_a, "min_value"_a, "max_value"_a);

    // Device transfers.
    image.def("to",
              py::overload_cast<const core::Device &, bool>(&Image::To,
                                                            py::const_),
              "Transfer the Image to a specified device.  A new image is "
              "always created if copy is true, else it is avoided when the "
              "original image is already on the target device.",
              "device"_a, "copy"_a = false);
    image.def("clone", &Image::Clone,
              "Returns a copy of the Image on the same device.");
    image.def(
            "cpu",
            [](const Image &image) { return image.To(core::Device("CPU:0")); },
            "Transfer the image to CPU. If the image "
            "is already on CPU, no copy will be performed.");
    image.def(
            "cuda",
            [](const Image &image, int device_id) {
                return image.To(core::Device("CUDA", device_id));
            },
            "Transfer the image to a CUDA device. If the image is already "
            "on the specified CUDA device, no copy will be performed.",
            "device_id"_a = 0);

    // Conversion.
    image.def("to",
              py::overload_cast<core::Dtype, bool, utility::optional<double>,
                                double>(&Image::To, py::const_),
              "Returns an Image with the specified Dtype.", "dtype"_a,
              "scale"_a = py::none(), "offset"_a = 0.0, "copy"_a = false);
    docstring::ClassMethodDocInject(
            m, "Image", "to",
            {{"dtype", "The targeted dtype to convert to."},
             {"scale",
              "Optional scale value. This is 1./255 for UInt8 -> Float{32,64}, "
              "1./65535 for UInt16 -> Float{32,64} and 1 otherwise"},
             {"offset", "Optional shift value. Default 0."},
             {"copy",
              "If true, a new tensor is always created; if false, the copy is "
              "avoided when the original tensor already has the targeted "
              "dtype."}});
    image.def("to_legacy", &Image::ToLegacy, "Convert to legacy Image type.");
    image.def_static("from_legacy", &Image::FromLegacy, "image_legacy"_a,
                     "device"_a = core::Device("CPU:0"),
                     "Create a Image from a legacy Open3D Image.");
    image.def("as_tensor", &Image::AsTensor);

    docstring::ClassMethodDocInject(m, "Image", "get_min_bound");
    docstring::ClassMethodDocInject(m, "Image", "get_max_bound");
    docstring::ClassMethodDocInject(m, "Image", "clear");
    docstring::ClassMethodDocInject(m, "Image", "is_empty");
    docstring::ClassMethodDocInject(m, "Image", "to_legacy");

    py::class_<RGBDImage, PyGeometry<RGBDImage>, std::shared_ptr<RGBDImage>,
               Geometry>
            rgbd_image(
                    m, "RGBDImage",
                    "RGBDImage is a pair of color and depth images. For most "
                    "processing, the image pair should be aligned (same "
                    "viewpoint and  "
                    "resolution).");
    rgbd_image
            // Constructors.
            .def(py::init<>(), "Construct an empty RGBDImage.")
            .def(py::init<const Image &, const Image &, bool>(),
                 "Parameterized constructor", "color"_a, "depth"_a,
                 "aligned"_a = true)
            // Depth and color images.
            .def_readwrite("color", &RGBDImage::color_, "The color image.")
            .def_readwrite("depth", &RGBDImage::depth_, "The depth image.")
            .def_readwrite("aligned_", &RGBDImage::aligned_,
                           "Are the depth and color images aligned (same "
                           "viewpoint and resolution)?")
            // Functions.
            .def("clear", &RGBDImage::Clear, "Clear stored data.")
            .def("is_empty", &RGBDImage::IsEmpty, "Is any data stored?")
            .def("are_aligned", &RGBDImage::AreAligned,
                 "Are the depth and color images aligned (same viewpoint and "
                 "resolution)?")
            .def("get_min_bound", &RGBDImage::GetMinBound,
                 "Compute min 2D coordinates for the data (always {0, 0}).")
            .def("get_max_bound", &RGBDImage::GetMaxBound,
                 "Compute max 2D coordinates for the data.")
            // Device transfers.
            .def("to",
                 py::overload_cast<const core::Device &, bool>(&RGBDImage::To,
                                                               py::const_),
                 "Transfer the RGBDImage to a specified device.", "device"_a,
                 "copy"_a = false)
            .def("clone", &RGBDImage::Clone,
                 "Returns a copy of the RGBDImage on the same device.")
            .def(
                    "cpu",
                    [](const RGBDImage &rgbd_image) {
                        return rgbd_image.To(core::Device("CPU:0"));
                    },
                    "Transfer the RGBD image to CPU. If the RGBD image "
                    "is already on CPU, no copy will be performed.")
            .def(
                    "cuda",
                    [](const RGBDImage &rgbd_image, int device_id) {
                        return rgbd_image.To(core::Device("CUDA", device_id));
                    },
                    "Transfer the RGBD image to a CUDA device. If the RGBD "
                    "image is already "
                    "on the specified CUDA device, no copy will be performed.",
                    "device_id"_a = 0)

            // Conversion.
            .def("to_legacy", &RGBDImage::ToLegacy,
                 "Convert to legacy RGBDImage type.")
            // Description.
            .def("__repr__", &RGBDImage::ToString);

    docstring::ClassMethodDocInject(m, "RGBDImage", "get_min_bound");
    docstring::ClassMethodDocInject(m, "RGBDImage", "get_max_bound");
    docstring::ClassMethodDocInject(m, "RGBDImage", "clear");
    docstring::ClassMethodDocInject(m, "RGBDImage", "is_empty");
    docstring::ClassMethodDocInject(m, "RGBDImage", "to_legacy");
    docstring::ClassMethodDocInject(m, "RGBDImage", "__init__",
                                    map_shared_argument_docstrings);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
