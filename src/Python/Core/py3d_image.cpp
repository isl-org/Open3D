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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Core.h>
using namespace three;

void pybind_image(py::module &m)
{
	py::class_<Image, PyGeometry2D<Image>, std::shared_ptr<Image>, Geometry2D>
			image(m, "Image", py::buffer_protocol());
	image
		.def(py::init<>())
		.def("__init__", [](Image &img, py::buffer b) {
			py::buffer_info info = b.request();
			int width, height, num_of_channels, bytes_per_channel;
			if (info.format == py::format_descriptor<uint8_t>::format() ||
					info.format == py::format_descriptor<int8_t>::format()) {
				bytes_per_channel = 1;
			} else if (info.format ==
					py::format_descriptor<uint16_t>::format() ||
					info.format == py::format_descriptor<int16_t>::format()) {
				bytes_per_channel = 2;
			} else if (info.format == py::format_descriptor<float>::format()) {
				bytes_per_channel = 4;
			} else {
				throw std::runtime_error("Image can only be initialized from buffer of uint8, uint16, or float!");
			}
			if (info.strides[info.ndim - 1] != bytes_per_channel) {
				throw std::runtime_error("Image can only be initialized from c-style buffer.");
			}
			if (info.ndim == 2) {
				num_of_channels = 1;
			} else if (info.ndim == 3) {
				num_of_channels = info.shape[2];
			}
			height = info.shape[0]; width = info.shape[1];
			img.PrepareImage(width, height, num_of_channels, bytes_per_channel);
			memcpy(img.data_.data(), info.ptr, img.data_.size());
		})
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
				throw std::runtime_error("Image has unrecognized bytes_per_channel.");
				break;
			}
			return py::buffer_info(
					img.data_.data(), img.bytes_per_channel_, format,
					3, {static_cast<unsigned long>(img.height_),
					static_cast<unsigned long>(img.width_),
					static_cast<unsigned long>(img.num_of_channels_)},
					{static_cast<unsigned long>(img.bytes_per_channel_ *
					img.num_of_channels_ * img.width_),
					static_cast<unsigned long>(img.bytes_per_channel_ *
					img.num_of_channels_),
					static_cast<unsigned long>(img.bytes_per_channel_)});
		})
		.def("__repr__", [](Image &img) {
			return std::string("Image of size ") + std::to_string(img.width_) +
					std::string("x") + std::to_string(img.height_) + ", with " +
					std::to_string(img.num_of_channels_) +
					std::string(" channels.\nUse numpy.asarray to access buffer data");
		});
}

void pybind_image_methods(py::module &m)
{
	m.def("CreateImageFromFile", &CreateImageFromFile,
			"Factory function to create an image from a file",
			"filename"_a);
}
