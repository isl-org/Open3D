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

#include <memory>

#include "open3d/geometry/RGBDImage.h"
#include "open3d/t/io/sensor/RGBDVideoReader.h"
#include "open3d/t/io/sensor/realsense/RSBagReader.h"
#include "pybind/docstring.h"
#include "pybind/t/io/io.h"

namespace open3d {
namespace t {
namespace io {

void pybind_sensor(py::module &m) {
    static const std::unordered_map<std::string, std::string>
            map_shared_argument_docstrings = {
                    {"timestamp", "Timestamp in the video (usec)."},
                    {"filename", "Path to the RGBD video file."},
                    {"frame_path",
                     "Frames will be stored in stream subfolders 'color' and "
                     "'depth' here. The intrinsic camera calibration for the "
                     "color stream will be saved in 'intrinsic.json'"},
                    {"start_time_us",
                     "(default 0) Start saving frames from this time (us)"},
                    {"end_time_us",
                     "(default video length) Save frames till this time (us)"}};

    py::enum_<SensorType>(m, "SensorType")
            .value("AZURE_KINECT", SensorType::AZURE_KINECT)
            .value("REAL_SENSE", SensorType::REAL_SENSE);

    // Class RGBD video metadata
    py::class_<RGBDVideoMetadata> rgbd_video_metadata(m, "RGBDVideoMetadata",
                                                      "RGBD Video metadata.");
    rgbd_video_metadata.def(py::init<>())
            .def_readwrite("width", &RGBDVideoMetadata::width_,
                           "Width of the video")
            .def_readwrite("height", &RGBDVideoMetadata::height_,
                           "Height of the video")
            .def_readwrite("fps", &RGBDVideoMetadata::fps_, "Video frame rate")
            .def_readwrite("color_format", &RGBDVideoMetadata::color_format_,
                           "Pixel format for color data")
            .def_readwrite("depth_format", &RGBDVideoMetadata::depth_format_,
                           "Pixel format for depth data")
            .def_readwrite("device_name", &RGBDVideoMetadata::device_name_,
                           "Capture device name")
            .def_readwrite("serial_number", &RGBDVideoMetadata::serial_number_,
                           "Capture device serial number")
            .def_readwrite("stream_length_usec",
                           &RGBDVideoMetadata::stream_length_usec_,
                           "Length of the video (usec)");

    // Class RGBD video reader
    py::class_<RGBDVideoReader, std::shared_ptr<RGBDVideoReader>>
            rgbd_video_reader(m, "RGBDVideoReader", "RGBD Video file reader.");
    rgbd_video_reader.def_static("create", &RGBDVideoReader::Create,
                                 "filename"_a,
                                 "Create RGBD video reader based on filename");
    docstring::ClassMethodDocInject(m, "RGBDVideoReader", "create",
                                    map_shared_argument_docstrings);

#ifdef BUILD_LIBREALSENSE
    // Class RS bag reader
    py::class_<RSBagReader, std::shared_ptr<RSBagReader>, RGBDVideoReader>
            rs_bag_reader(m, "RSBagReader", "RealSense Bag file reader.");
    rs_bag_reader.def(py::init<>())
            .def("is_opened", &RSBagReader::IsOpened,
                 "Check if the RS bag file  is opened.")
            .def("open", &RSBagReader::Open, "filename"_a,
                 "Open an RS bag playback.")
            .def("close", &RSBagReader::Close,
                 "Close the opened RS bag playback.")
            .def("is_eof", &RSBagReader::IsEOF,
                 "Check if the RS bag file is all read.")
            .def("get_metadata", &RSBagReader::GetMetadata,
                 "Get metadata of the RS bag playback.")
            .def("seek_timestamp", &RSBagReader::SeekTimestamp, "timestamp"_a,
                 "Seek to the timestamp (in us).")
            .def("get_timestamp", &RSBagReader::GetTimestamp,
                 "Get current timestamp (in us).")
            .def("next_frame", &RSBagReader::NextFrame,
                 "Get next frame from the RS bag playback and returns the RGBD "
                 "object.")
            .def("save_frames", &RSBagReader::SaveFrames,
                 py::call_guard<py::gil_scoped_release>(), "frame_path"_a,
                 "start_time_us"_a = 0, "end_time_us"_a = UINT64_MAX,
                 "Save synchronized and aligned individual frames to "
                 "subfolders");
    docstring::ClassMethodDocInject(m, "RSBagReader", "open",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RSBagReader", "seek_timestamp",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RSBagReader", "save_frames",
                                    map_shared_argument_docstrings);
#endif
}

}  // namespace io
}  // namespace t
}  // namespace open3d
