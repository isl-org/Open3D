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

#include <Python/docstring.h>
#include <json/json.h>
#include "Python/io/io.h"

#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/IO/Sensor/AzureKinect/AzureKinectRecorder.h"
#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensor.h"
#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensorConfig.h"
#include "Open3D/IO/Sensor/AzureKinect/MKVReader.h"

using namespace open3d;

void pybind_sensor(py::module &m) {
    static const std::unordered_map<std::string, std::string>
            map_shared_argument_docstrings = {
                    {"sensor_index", "The selected device index."},
                    {"config", "AzureKinectSensor's config file."},
                    {"timestamp", "Timestamp in the video (usec)."},
                    {"filename", "Path to the mkv file."},
                    {"enable_record", "Enable recording to mkv file."},
                    {"enable_align_depth_to_color",
                     "Enable aligning WFOV depth image to the color image in "
                     "visualizer."}};

    // Class kinect config
    py::class_<io::AzureKinectSensorConfig> azure_kinect_sensor_config(
            m, "AzureKinectSensorConfig", "AzureKinect sensor configuration.");
    py::detail::bind_default_constructor<io::AzureKinectSensorConfig>(
            azure_kinect_sensor_config);
    azure_kinect_sensor_config.def(
            py::init([](const std::unordered_map<std::string, std::string>
                                &config) {
                return new io::AzureKinectSensorConfig(config);
            }),
            "config"_a);

    py::class_<io::MKVMetadata> azure_kinect_mkv_metadata(
            m, "AzureKinectMKVMetadata", "AzureKinect mkv metadata.");
    py::detail::bind_default_constructor<io::MKVMetadata>(
            azure_kinect_mkv_metadata);
    azure_kinect_mkv_metadata
            .def_readwrite("width", &io::MKVMetadata::width_,
                           "Width of the video")
            .def_readwrite("height", &io::MKVMetadata::height_,
                           "Height of the video")
            .def_readwrite("stream_length_usec",
                           &io::MKVMetadata::stream_length_usec_,
                           "Length of the video (usec)");

    // Class sensor
    py::class_<io::AzureKinectSensor> azure_kinect_sensor(
            m, "AzureKinectSensor", "AzureKinect sensor.");

    azure_kinect_sensor.def(
            py::init([](const io::AzureKinectSensorConfig &sensor_config) {
                return new io::AzureKinectSensor(sensor_config);
            }),
            "sensor_config"_a);
    azure_kinect_sensor
            .def("connect", &io::AzureKinectSensor::Connect, "sensor_index"_a,
                 "Connect to specified device.")
            .def("capture_frame", &io::AzureKinectSensor::CaptureFrame,
                 "enable_align_depth_to_color"_a, "Capture an RGBD frame.")
            .def_static("list_devices", &io::AzureKinectSensor::ListDevices,
                        "List available Azure Kinect devices");
    docstring::ClassMethodDocInject(m, "AzureKinectSensor", "connect",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectSensor", "capture_frame",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectSensor", "list_devices",
                                    map_shared_argument_docstrings);

    // Class recorder
    py::class_<io::AzureKinectRecorder> azure_kinect_recorder(
            m, "AzureKinectRecorder", "AzureKinect recorder.");

    azure_kinect_recorder.def(
            py::init([](const io::AzureKinectSensorConfig &sensor_config,
                        size_t sensor_index) {
                return new io::AzureKinectRecorder(sensor_config, sensor_index);
            }),
            "sensor_config"_a, "sensor_index"_a);
    azure_kinect_recorder
            .def("init_sensor", &io::AzureKinectRecorder::InitSensor,
                 "Initialize sensor.")
            .def("is_record_created", &io::AzureKinectRecorder::IsRecordCreated,
                 "Check if the mkv file is created.")
            .def("open_record", &io::AzureKinectRecorder::OpenRecord,
                 "filename"_a, "Attempt to create and open an mkv file.")
            .def("close_record", &io::AzureKinectRecorder::CloseRecord,
                 "Close the recorded mkv file.")
            .def("record_frame", &io::AzureKinectRecorder::RecordFrame,
                 "enable_record"_a, "enable_align_depth_to_color"_a,
                 "Record a frame to mkv if flag is on and return an RGBD "
                 "object.");
    docstring::ClassMethodDocInject(m, "AzureKinectRecorder", "init_sensor",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectRecorder",
                                    "is_record_created",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectRecorder", "open_record",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectRecorder", "close_record",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectRecorder", "record_frame",
                                    map_shared_argument_docstrings);

    // Class mkv reader
    py::class_<io::MKVReader> azure_kinect_mkv_reader(
            m, "AzureKinectMKVReader", "AzureKinect mkv file reader.");
    azure_kinect_mkv_reader.def(py::init([]() { return io::MKVReader(); }));
    azure_kinect_mkv_reader
            .def("is_opened", &io::MKVReader::IsOpened, "Is mkv file opened.")
            .def("open", &io::MKVReader::Open, "filename"_a,
                 "Open an mkv playback.")
            .def("close", &io::MKVReader::Close,
                 "Close the opened mkv playback.")
            .def("is_eof", &io::MKVReader::IsEOF,
                 "Check if the mkv file is all read.")
            .def("get_metadata", &io::MKVReader::GetMetadata,
                 "Get metadata of the mkv playback.")
            .def("seek_timestamp", &io::MKVReader::SeekTimestamp, "timestamp"_a,
                 "Seek to the timestamp (in us).")
            .def("next_frame", &io::MKVReader::NextFrame,
                 "Get next frame from the mkv playback and returns the RGBD "
                 "object.");
    docstring::ClassMethodDocInject(m, "AzureKinectMKVReader", "open",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectMKVReader", "close",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectMKVReader", "is_eof",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectMKVReader", "get_metadata",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectMKVReader", "seek_timestamp",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectMKVReader", "next_frame",
                                    map_shared_argument_docstrings);
}
