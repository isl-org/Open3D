// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/RGBDImage.h"
#include "open3d/io/sensor/azure_kinect/AzureKinectRecorder.h"
#include "open3d/io/sensor/azure_kinect/AzureKinectSensor.h"
#include "open3d/io/sensor/azure_kinect/AzureKinectSensorConfig.h"
#include "open3d/io/sensor/azure_kinect/MKVReader.h"
#include "pybind/docstring.h"
#include "pybind/io/io.h"

namespace open3d {
namespace io {

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
    py::class_<AzureKinectSensorConfig> azure_kinect_sensor_config(
            m, "AzureKinectSensorConfig", "AzureKinect sensor configuration.");
    py::detail::bind_default_constructor<AzureKinectSensorConfig>(
            azure_kinect_sensor_config);
    azure_kinect_sensor_config.def(
            py::init([](const std::unordered_map<std::string, std::string>
                                &config) {
                return new AzureKinectSensorConfig(config);
            }),
            "config"_a);

    py::class_<MKVMetadata> azure_kinect_mkv_metadata(
            m, "AzureKinectMKVMetadata", "AzureKinect mkv metadata.");
    py::detail::bind_default_constructor<MKVMetadata>(
            azure_kinect_mkv_metadata);
    azure_kinect_mkv_metadata
            .def_readwrite("width", &MKVMetadata::width_, "Width of the video")
            .def_readwrite("height", &MKVMetadata::height_,
                           "Height of the video")
            .def_readwrite("stream_length_usec",
                           &MKVMetadata::stream_length_usec_,
                           "Length of the video (usec)");

    // Class sensor
    py::class_<AzureKinectSensor> azure_kinect_sensor(m, "AzureKinectSensor",
                                                      "AzureKinect sensor.");

    azure_kinect_sensor.def(
            py::init([](const AzureKinectSensorConfig &sensor_config) {
                return new AzureKinectSensor(sensor_config);
            }),
            "sensor_config"_a);
    azure_kinect_sensor
            .def("connect", &AzureKinectSensor::Connect, "sensor_index"_a,
                 "Connect to specified device.")
            .def("disconnect", &AzureKinectSensor::Disconnect,
                 "Disconnect from the connected device.")
            .def("capture_frame", &AzureKinectSensor::CaptureFrame,
                 "enable_align_depth_to_color"_a, "Capture an RGBD frame.")
            .def_static("list_devices", &AzureKinectSensor::ListDevices,
                        "List available Azure Kinect devices");
    docstring::ClassMethodDocInject(m, "AzureKinectSensor", "connect",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectSensor", "capture_frame",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "AzureKinectSensor", "list_devices",
                                    map_shared_argument_docstrings);

    // Class recorder
    py::class_<AzureKinectRecorder> azure_kinect_recorder(
            m, "AzureKinectRecorder", "AzureKinect recorder.");

    azure_kinect_recorder.def(
            py::init([](const AzureKinectSensorConfig &sensor_config,
                        size_t sensor_index) {
                return new AzureKinectRecorder(sensor_config, sensor_index);
            }),
            "sensor_config"_a, "sensor_index"_a);
    azure_kinect_recorder
            .def("init_sensor", &AzureKinectRecorder::InitSensor,
                 "Initialize sensor.")
            .def("is_record_created", &AzureKinectRecorder::IsRecordCreated,
                 "Check if the mkv file is created.")
            .def("open_record", &AzureKinectRecorder::OpenRecord, "filename"_a,
                 "Attempt to create and open an mkv file.")
            .def("close_record", &AzureKinectRecorder::CloseRecord,
                 "Close the recorded mkv file.")
            .def("record_frame", &AzureKinectRecorder::RecordFrame,
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
    py::class_<MKVReader> azure_kinect_mkv_reader(
            m, "AzureKinectMKVReader", "AzureKinect mkv file reader.");
    azure_kinect_mkv_reader.def(py::init([]() { return MKVReader(); }));
    azure_kinect_mkv_reader
            .def("is_opened", &MKVReader::IsOpened,
                 "Check if the mkv file  is opened.")
            .def("open", &MKVReader::Open, "filename"_a,
                 "Open an mkv playback.")
            .def("close", &MKVReader::Close, "Close the opened mkv playback.")
            .def("is_eof", &MKVReader::IsEOF,
                 "Check if the mkv file is all read.")
            .def("get_metadata", &MKVReader::GetMetadata,
                 "Get metadata of the mkv playback.")
            .def("seek_timestamp", &MKVReader::SeekTimestamp, "timestamp"_a,
                 "Seek to the timestamp (in us).")
            .def("next_frame", &MKVReader::NextFrame,
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

}  // namespace io
}  // namespace open3d
