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
#include "open3d/t/io/sensor/RGBDSensor.h"
#include "open3d/t/io/sensor/RGBDVideoReader.h"
#ifdef BUILD_LIBREALSENSE
#include "open3d/t/io/sensor/realsense/RSBagReader.h"
#include "open3d/t/io/sensor/realsense/RealSenseSensor.h"
#include "open3d/t/io/sensor/realsense/RealSenseSensorConfig.h"
#endif
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
                     "(default video length) Save frames till this time (us)"},
                    {"buffer_size",
                     "Size of internal frame buffer, increase this if you "
                     "experience frame drops."}};

    py::enum_<SensorType>(m, "SensorType", "Sensor type")
            .value("AZURE_KINECT", SensorType::AZURE_KINECT)
            .value("REAL_SENSE", SensorType::REAL_SENSE);

    // Class RGBD video metadata
    py::class_<RGBDVideoMetadata> rgbd_video_metadata(m, "RGBDVideoMetadata",
                                                      "RGBD Video metadata.");
    rgbd_video_metadata.def(py::init<>())
            .def_readwrite("intrinsics", &RGBDVideoMetadata::intrinsics_,
                           "Shared intrinsics between RGB & depth")
            .def_readwrite("device_name", &RGBDVideoMetadata::device_name_,
                           "Capture device name")
            .def_readwrite("serial_number", &RGBDVideoMetadata::serial_number_,
                           "Capture device serial number")
            .def_readwrite("stream_length_usec",
                           &RGBDVideoMetadata::stream_length_usec_,
                           "Length of the video (usec). 0 for live capture.")
            .def_readwrite("width", &RGBDVideoMetadata::width_,
                           "Width of the video")
            .def_readwrite("height", &RGBDVideoMetadata::height_,
                           "Height of the video")
            .def_readwrite("fps", &RGBDVideoMetadata::fps_,
                           "Video frame rate (common for both color and depth)")
            .def_readwrite("color_format", &RGBDVideoMetadata::color_format_,
                           "Pixel format for color data")
            .def_readwrite("color_dt", &RGBDVideoMetadata::color_dt_,
                           "Pixel Dtype for color data.")
            .def_readwrite("depth_format", &RGBDVideoMetadata::depth_format_,
                           "Pixel format for depth data")
            .def_readwrite("depth_dt", &RGBDVideoMetadata::depth_dt_,
                           "Pixel Dtype for depth data.")
            .def_readwrite("depth_scale", &RGBDVideoMetadata::depth_scale_,
                           "Number of depth units per meter (depth in m = "
                           "depth_pixel_value/depth_scale).")
            .def_readwrite("color_channels",
                           &RGBDVideoMetadata::color_channels_,
                           "Number of color channels.")
            .def("__repr__", &RGBDVideoMetadata::ToString);

    // RGBD video reader trampoline
    class PyRGBDVideoReader : public RGBDVideoReader {
    public:
        using RGBDVideoReader::RGBDVideoReader;
        bool IsOpened() const override {
            PYBIND11_OVERLOAD_PURE(bool, RGBDVideoReader, );
        }

        bool IsEOF() const override {
            PYBIND11_OVERLOAD_PURE(bool, RGBDVideoReader, );
        }

        bool Open(const std::string &filename) override {
            PYBIND11_OVERLOAD_PURE(bool, RGBDVideoReader, Open, filename);
        }

        void Close() override {
            PYBIND11_OVERLOAD_PURE(void, RGBDVideoReader, );
        }

        RGBDVideoMetadata &GetMetadata() override {
            PYBIND11_OVERLOAD_PURE(RGBDVideoMetadata &, RGBDVideoReader, );
        }

        const RGBDVideoMetadata &GetMetadata() const override {
            PYBIND11_OVERLOAD_PURE(const RGBDVideoMetadata &,
                                   RGBDVideoReader, );
        }

        bool SeekTimestamp(uint64_t timestamp) override {
            PYBIND11_OVERLOAD_PURE(bool, RGBDVideoReader, SeekTimestamp,
                                   timestamp);
        }

        uint64_t GetTimestamp() const override {
            PYBIND11_OVERLOAD_PURE(uint64_t, RGBDVideoReader, );
        }

        t::geometry::RGBDImage NextFrame() override {
            PYBIND11_OVERLOAD_PURE(t::geometry::RGBDImage, RGBDVideoReader, );
        }

        std::string GetFilename() const override {
            PYBIND11_OVERLOAD_PURE(std::string, RGBDVideoReader, );
        }
    };

    // Class RGBD video reader
    py::class_<RGBDVideoReader, PyRGBDVideoReader,
               std::unique_ptr<RGBDVideoReader>>
            rgbd_video_reader(m, "RGBDVideoReader", "RGBD Video file reader.");
    rgbd_video_reader.def(py::init<>())
            .def_static("create", &RGBDVideoReader::Create, "filename"_a,
                        "Create RGBD video reader based on filename")
            .def("save_frames", &RGBDVideoReader::SaveFrames, "frame_path"_a,
                 "start_time_us"_a = 0, "end_time_us"_a = UINT64_MAX,
                 "Save synchronized and aligned individual frames to "
                 "subfolders.")
            .def("__repr__", &RGBDVideoReader::ToString);
    docstring::ClassMethodDocInject(m, "RGBDVideoReader", "create",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RGBDVideoReader", "save_frames",
                                    map_shared_argument_docstrings);

    // Class RGBD sensor
    py::class_<RGBDSensor> rgbd_sensor(
            m, "RGBDSensor", "Interface class for control of RGBD cameras.");
    rgbd_sensor.def("__repr__", &RGBDSensor::ToString);

#ifdef BUILD_LIBREALSENSE
    // Class RS bag reader
    py::class_<RSBagReader, std::unique_ptr<RSBagReader>, RGBDVideoReader>
            rs_bag_reader(
                    m, "RSBagReader",
                    "RealSense Bag file reader."
                    "Only the first color and depth streams from the bag file "
                    "will be read.\n"
                    " - The streams must have the same frame rate.\n"
                    " - The color stream must have RGB 8 bit (RGB8/BGR8) pixel "
                    "format\n"
                    " - The depth stream must have 16 bit unsigned int (Z16) "
                    "pixel format\n"
                    "The output is synchronized color and depth frame pairs "
                    "with the depth frame aligned to the color frame."
                    "Unsynchronized frames will be dropped. With alignment,"
                    "the depth and color frames have the same  viewpoint and"
                    "resolution. See "
                    "https://intelrealsense.github.io/librealsense/doxygen/"
                    "rs__sensor_8h.html#ae04b7887ce35d16dbd9d2d295d23aac7"
                    "for format documentation\n"
                    "Note: A few frames may be dropped if user code takes a "
                    "long time (>10 frame intervals) to process a frame.");
    rs_bag_reader.def(py::init<>())
            .def(py::init<size_t>(),
                 "buffer_size"_a = RSBagReader::DEFAULT_BUFFER_SIZE)
            .def("is_opened", &RSBagReader::IsOpened,
                 "Check if the RS bag file  is opened.")
            .def("open", &RSBagReader::Open, "filename"_a,
                 "Open an RS bag playback.")
            .def("close", &RSBagReader::Close,
                 "Close the opened RS bag playback.")
            .def("is_eof", &RSBagReader::IsEOF,
                 "Check if the RS bag file is all read.")
            .def_property(
                    "metadata",
                    py::overload_cast<>(&RSBagReader::GetMetadata, py::const_),
                    py::overload_cast<>(&RSBagReader::GetMetadata),
                    "Get metadata of the RS bag playback.")
            .def("seek_timestamp", &RSBagReader::SeekTimestamp, "timestamp"_a,
                 "Seek to the timestamp (in us).")
            .def("get_timestamp", &RSBagReader::GetTimestamp,
                 "Get current timestamp (in us).")
            .def("next_frame", &RSBagReader::NextFrame,
                 "Get next frame from the RS bag playback and returns the RGBD "
                 "object.")
            // Release Python GIL for SaveFrames, since this will take a while
            .def("save_frames", &RSBagReader::SaveFrames,
                 py::call_guard<py::gil_scoped_release>(), "frame_path"_a,
                 "start_time_us"_a = 0, "end_time_us"_a = UINT64_MAX,
                 "Save synchronized and aligned individual frames to "
                 "subfolders")
            .def("__repr__", &RSBagReader::ToString);
    docstring::ClassMethodDocInject(m, "RSBagReader", "__init__",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RSBagReader", "open",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RSBagReader", "seek_timestamp",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "RSBagReader", "save_frames",
                                    map_shared_argument_docstrings);

    // Class RealSenseSensorConfig
    py::class_<RealSenseSensorConfig> realsense_sensor_config(
            m, "RealSenseSensorConfig", "Configuration for a RealSense camera");

    realsense_sensor_config.def(py::init<>(), "Default config will be used")
            .def(py::init<const std::unordered_map<std::string, std::string>
                                  &>(),
                 "config"_a, "Initialize config with a map");

    py::class_<RealSenseValidConfigs> realsense_valid_configs(
            m, "RealSenseValidConfigs",
            "Store set of valid configuration options for a connected "
            "RealSense device.  From this structure, a user can construct a "
            "RealSenseSensorConfig object meeting their specifications.");
    realsense_valid_configs
            .def_readwrite("serial", &RealSenseValidConfigs::serial,
                           "Device serial number.")
            .def_readwrite("name", &RealSenseValidConfigs::name, "Device name.")
            .def_readwrite("valid_configs",
                           &RealSenseValidConfigs::valid_configs,
                           "Mapping between configuraiton option name and a "
                           "list of valid values.");

    // Class RealSenseSensor
    py::class_<RealSenseSensor, RGBDSensor> realsense_sensor(
            m, "RealSenseSensor",
            "RealSense camera discovery, configuration, streaming and "
            "recording");
    realsense_sensor.def(py::init<>(), "Initialize with default settings.")
            .def_static("list_devices", &RealSenseSensor::ListDevices,
                        "List all RealSense cameras connected to the system "
                        "along with their capabilities. Use this listing to "
                        "select an appropriate configuration for a camera")
            .def_static("enumerate_devices", &RealSenseSensor::EnumerateDevices,
                        "Query all connected RealSense cameras for their "
                        "capabilities.")
            .def("init_sensor",
                 py::overload_cast<const RGBDSensorConfig &, size_t,
                                   const std::string &>(
                         &RealSenseSensor::InitSensor),
                 "sensor_config"_a = RealSenseSensorConfig{},
                 "sensor_index"_a = 0, "filename"_a = "",
                 "Configure sensor with custom settings. If this is skipped, "
                 "default settings will be used. You can enable recording to a "
                 "bag file by specifying a filename.")
            .def("init_sensor",
                 py::overload_cast<const RealSenseSensorConfig &, size_t,
                                   const std::string &>(
                         &RealSenseSensor::InitSensor),
                 "sensor_config"_a = RealSenseSensorConfig{},
                 "sensor_index"_a = 0, "filename"_a = "",
                 "Configure sensor with custom settings. If this is skipped, "
                 "default settings will be used. You can enable recording to a "
                 "bag file by specifying a filename.")
            .def("start_capture", &RealSenseSensor::StartCapture,
                 "start_record"_a = false,
                 "Start capturing synchronized depth and color frames.")
            .def("pause_record", &RealSenseSensor::PauseRecord,
                 "Pause recording to the bag file. Note: If this is called "
                 "immediately after start_capture, the bag file may have an "
                 "incorrect end time.")
            .def("resume_record", &RealSenseSensor::ResumeRecord,
                 "Resume recording to the bag file. The file will contain "
                 "discontinuous segments.")
            .def("capture_frame", &RealSenseSensor::CaptureFrame,
                 "wait"_a = true, "align_depth_to_color"_a = true,
                 "Acquire the next synchronized RGBD frameset from the camera.")
            .def("get_timestamp", &RealSenseSensor::GetTimestamp,
                 "Get current timestamp (in us)")
            .def("stop_capture", &RealSenseSensor::StopCapture,
                 "Stop capturing frames.")
            .def("get_metadata", &RealSenseSensor::GetMetadata,
                 "Get metadata of the RealSense video capture.")
            .def("get_filename", &RealSenseSensor::GetFilename,
                 "Get filename being written.")
            .def("__repr__", &RealSenseSensor::ToString);

    docstring::ClassMethodDocInject(
            m, "RealSenseSensor", "init_sensor",
            {{"sensor_config",
              "Camera configuration, such as resolution and framerate. A "
              "serial number can be entered here to connect to a specific "
              "camera."},
             {"sensor_index",
              "Connect to a camera at this position in the enumeration of "
              "RealSense cameras that are currently connected. Use "
              "enumerate_devices() or list_devices() to obtain a list of "
              "connected cameras. This is ignored if sensor_config contains "
              "a serial entry."},
             {"filename", "Save frames to a bag file"}});
    docstring::ClassMethodDocInject(
            m, "RealSenseSensor", "start_capture",
            {{"start_record",
              "Start recording to the specified bag file as well."}});
    docstring::ClassMethodDocInject(
            m, "RealSenseSensor", "capture_frame",
            {{"wait",
              "If true wait for the next frame set, else return immediately "
              "with an empty RGBDImage if it is not yet available."},
             {"align_depth_to_color",
              "Enable aligning WFOV depth image to the color image in "
              "visualizer."}});

#endif
}

}  // namespace io
}  // namespace t
}  // namespace open3d
