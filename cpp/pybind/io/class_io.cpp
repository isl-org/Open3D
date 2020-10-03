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

#include <string>
#include <unordered_map>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/io/FeatureIO.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/LineSetIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/io/PoseGraphIO.h"
#include "open3d/io/TPointCloudIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/io/VoxelGridIO.h"
#include "pybind/docstring.h"
#include "pybind/io/io.h"

#ifdef BUILD_AZURE_KINECT
#include "open3d/io/sensor/azure_kinect/AzureKinectSensorConfig.h"
#include "open3d/io/sensor/azure_kinect/MKVMetadata.h"
#endif

namespace open3d {
namespace io {

// IO functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"filename", "Path to file."},
                // Write options
                {"compressed",
                 "Set to ``True`` to write in compressed format."},
                {"format",
                 "The format of the input file. When not specified or set as "
                 "``auto``, the format is inferred from file extension name."},
                {"remove_nan_points",
                 "If true, all points that include a NaN are removed from "
                 "the PointCloud."},
                {"remove_infinite_points",
                 "If true, all points that include an infinite value are "
                 "removed from the PointCloud."},
                {"quality", "Quality of the output file."},
                {"write_ascii",
                 "Set to ``True`` to output in ascii format, otherwise binary "
                 "format will be used."},
                {"write_vertex_normals",
                 "Set to ``False`` to not write any vertex normals, even if "
                 "present on the mesh"},
                {"write_vertex_colors",
                 "Set to ``False`` to not write any vertex colors, even if "
                 "present on the mesh"},
                {"write_triangle_uvs",
                 "Set to ``False`` to not write any triangle uvs, even if "
                 "present on the mesh. For ``obj`` format, mtl file is saved "
                 "only when ``True`` is set"},
                // Entities
                {"config", "AzureKinectSensor's config file."},
                {"pointcloud", "The ``PointCloud`` object for I/O"},
                {"mesh", "The ``TriangleMesh`` object for I/O"},
                {"line_set", "The ``LineSet`` object for I/O"},
                {"image", "The ``Image`` object for I/O"},
                {"voxel_grid", "The ``VoxelGrid`` object for I/O"},
                {"trajectory",
                 "The ``PinholeCameraTrajectory`` object for I/O"},
                {"intrinsic", "The ``PinholeCameraIntrinsic`` object for I/O"},
                {"parameters",
                 "The ``PinholeCameraParameters`` object for I/O"},
                {"pose_graph", "The ``PoseGraph`` object for I/O"},
                {"feature", "The ``Feature`` object for I/O"},
                {"print_progress",
                 "If set to true a progress bar is visualized in the console"},
};

void pybind_class_io(py::module &m_io) {
    py::enum_<FileGeometry> geom_type(m_io, "FileGeometry", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    geom_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Geometry types";
            }),
            py::none(), py::none(), "");
    geom_type.value("CONTENTS_UKNWOWN", FileGeometry::CONTENTS_UNKNOWN)
            .value("CONTAINS_POINTS", FileGeometry::CONTAINS_POINTS)
            .value("CONTAINS_LINES", FileGeometry::CONTAINS_LINES)
            .value("CONTAINS_TRIANGLES", FileGeometry::CONTAINS_TRIANGLES)
            .export_values();
    m_io.def(
            "read_file_geometry_type", &ReadFileGeometryType,
            "Returns the type of geometry of the file. This is a faster way of "
            "determining the file type than attempting to read the file as a "
            "point cloud, mesh, or line set in turn.");

    // open3d::geometry::Image
    m_io.def(
            "read_image",
            [](const std::string &filename) {
                py::gil_scoped_release release;
                geometry::Image image;
                ReadImage(filename, image);
                return image;
            },
            "Function to read Image from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_image",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_image",
            [](const std::string &filename, const geometry::Image &image,
               int quality) {
                py::gil_scoped_release release;
                return WriteImage(filename, image, quality);
            },
            "Function to write Image to file", "filename"_a, "image"_a,
            "quality"_a = kOpen3DImageIODefaultQuality);
    docstring::FunctionDocInject(m_io, "write_image",
                                 map_shared_argument_docstrings);

    // open3d::geometry::LineSet
    m_io.def(
            "read_line_set",
            [](const std::string &filename, const std::string &format,
               bool print_progress) {
                py::gil_scoped_release release;
                geometry::LineSet line_set;
                ReadLineSet(filename, line_set, format, print_progress);
                return line_set;
            },
            "Function to read LineSet from file", "filename"_a,
            "format"_a = "auto", "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_line_set",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_line_set",
            [](const std::string &filename, const geometry::LineSet &line_set,
               bool write_ascii, bool compressed, bool print_progress) {
                py::gil_scoped_release release;
                return WriteLineSet(filename, line_set, write_ascii, compressed,
                                    print_progress);
            },
            "Function to write LineSet to file", "filename"_a, "line_set"_a,
            "write_ascii"_a = false, "compressed"_a = false,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_line_set",
                                 map_shared_argument_docstrings);

    // open3d::geometry::PointCloud
    m_io.def(
            "read_point_cloud",
            [](const std::string &filename, const std::string &format,
               bool remove_nan_points, bool remove_infinite_points,
               bool print_progress) {
                py::gil_scoped_release release;
                geometry::PointCloud pcd;
                ReadPointCloud(filename, pcd,
                               {format, remove_nan_points,
                                remove_infinite_points, print_progress});
                return pcd;
            },
            "Function to read PointCloud from file", "filename"_a,
            "format"_a = "auto", "remove_nan_points"_a = true,
            "remove_infinite_points"_a = true, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_point_cloud",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_point_cloud",
            [](const std::string &filename,
               const geometry::PointCloud &pointcloud, bool write_ascii,
               bool compressed, bool print_progress) {
                py::gil_scoped_release release;
                return WritePointCloud(
                        filename, pointcloud,
                        {write_ascii, compressed, print_progress});
            },
            "Function to write PointCloud to file", "filename"_a,
            "pointcloud"_a, "write_ascii"_a = false, "compressed"_a = false,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_point_cloud",
                                 map_shared_argument_docstrings);

    // open3d::t::geometry::PointCloud
    m_io.def(
            "read_t_point_cloud",
            [](const std::string &filename, const std::string &format,
               bool remove_nan_points, bool remove_infinite_points,
               bool print_progress) {
                py::gil_scoped_release release;
                t::geometry::PointCloud pcd;
                ReadPointCloud(filename, pcd,
                               {format, remove_nan_points,
                                remove_infinite_points, print_progress});
                return pcd;
            },
            "Function to read PointCloud with tensor attributes from file",
            "filename"_a, "format"_a = "auto", "remove_nan_points"_a = false,
            "remove_infinite_points"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_point_cloud",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_t_point_cloud",
            [](const std::string &filename,
               const t::geometry::PointCloud &pointcloud, bool write_ascii,
               bool compressed, bool print_progress) {
                py::gil_scoped_release release;
                return WritePointCloud(
                        filename, pointcloud,
                        {write_ascii, compressed, print_progress});
            },
            "Function to write PointCloud with tensor attributes to file",
            "filename"_a, "pointcloud"_a, "write_ascii"_a = true,
            "compressed"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_point_cloud",
                                 map_shared_argument_docstrings);

    // open3d::geometry::TriangleMesh
    m_io.def(
            "read_triangle_mesh",
            [](const std::string &filename, bool print_progress) {
                py::gil_scoped_release release;
                geometry::TriangleMesh mesh;
                ReadTriangleMesh(filename, mesh, print_progress);
                return mesh;
            },
            "Function to read TriangleMesh from file", "filename"_a,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_triangle_mesh",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_triangle_mesh",
            [](const std::string &filename, const geometry::TriangleMesh &mesh,
               bool write_ascii, bool compressed, bool write_vertex_normals,
               bool write_vertex_colors, bool write_triangle_uvs,
               bool print_progress) {
                py::gil_scoped_release release;
                return WriteTriangleMesh(filename, mesh, write_ascii,
                                         compressed, write_vertex_normals,
                                         write_vertex_colors,
                                         write_triangle_uvs, print_progress);
            },
            "Function to write TriangleMesh to file", "filename"_a, "mesh"_a,
            "write_ascii"_a = false, "compressed"_a = false,
            "write_vertex_normals"_a = true, "write_vertex_colors"_a = true,
            "write_triangle_uvs"_a = true, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_triangle_mesh",
                                 map_shared_argument_docstrings);

    // open3d::geometry::VoxelGrid
    m_io.def(
            "read_voxel_grid",
            [](const std::string &filename, const std::string &format,
               bool print_progress) {
                py::gil_scoped_release release;
                geometry::VoxelGrid voxel_grid;
                ReadVoxelGrid(filename, voxel_grid, format);
                return voxel_grid;
            },
            "Function to read VoxelGrid from file", "filename"_a,
            "format"_a = "auto", "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_voxel_grid",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_voxel_grid",
            [](const std::string &filename,
               const geometry::VoxelGrid &voxel_grid, bool write_ascii,
               bool compressed, bool print_progress) {
                py::gil_scoped_release release;
                return WriteVoxelGrid(filename, voxel_grid, write_ascii,
                                      compressed, print_progress);
            },
            "Function to write VoxelGrid to file", "filename"_a, "voxel_grid"_a,
            "write_ascii"_a = false, "compressed"_a = false,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_voxel_grid",
                                 map_shared_argument_docstrings);

    // open3d::camera
    m_io.def(
            "read_pinhole_camera_intrinsic",
            [](const std::string &filename) {
                py::gil_scoped_release release;
                camera::PinholeCameraIntrinsic intrinsic;
                ReadIJsonConvertible(filename, intrinsic);
                return intrinsic;
            },
            "Function to read PinholeCameraIntrinsic from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pinhole_camera_intrinsic",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pinhole_camera_intrinsic",
            [](const std::string &filename,
               const camera::PinholeCameraIntrinsic &intrinsic) {
                py::gil_scoped_release release;
                return WriteIJsonConvertible(filename, intrinsic);
            },
            "Function to write PinholeCameraIntrinsic to file", "filename"_a,
            "intrinsic"_a);
    docstring::FunctionDocInject(m_io, "write_pinhole_camera_intrinsic",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_pinhole_camera_parameters",
            [](const std::string &filename) {
                py::gil_scoped_release release;
                camera::PinholeCameraParameters parameters;
                ReadIJsonConvertible(filename, parameters);
                return parameters;
            },
            "Function to read PinholeCameraParameters from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pinhole_camera_parameters",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pinhole_camera_parameters",
            [](const std::string &filename,
               const camera::PinholeCameraParameters &parameters) {
                py::gil_scoped_release release;
                return WriteIJsonConvertible(filename, parameters);
            },
            "Function to write PinholeCameraParameters to file", "filename"_a,
            "parameters"_a);
    docstring::FunctionDocInject(m_io, "write_pinhole_camera_parameters",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_pinhole_camera_trajectory",
            [](const std::string &filename) {
                py::gil_scoped_release release;
                camera::PinholeCameraTrajectory trajectory;
                ReadPinholeCameraTrajectory(filename, trajectory);
                return trajectory;
            },
            "Function to read PinholeCameraTrajectory from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pinhole_camera_trajectory",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pinhole_camera_trajectory",
            [](const std::string &filename,
               const camera::PinholeCameraTrajectory &trajectory) {
                py::gil_scoped_release release;
                return WritePinholeCameraTrajectory(filename, trajectory);
            },
            "Function to write PinholeCameraTrajectory to file", "filename"_a,
            "trajectory"_a);
    docstring::FunctionDocInject(m_io, "write_pinhole_camera_trajectory",
                                 map_shared_argument_docstrings);

    // open3d::registration
    m_io.def(
            "read_feature",
            [](const std::string &filename) {
                py::gil_scoped_release release;
                pipelines::registration::Feature feature;
                ReadFeature(filename, feature);
                return feature;
            },
            "Function to read registration.Feature from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_feature",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_feature",
            [](const std::string &filename,
               const pipelines::registration::Feature &feature) {
                py::gil_scoped_release release;
                return WriteFeature(filename, feature);
            },
            "Function to write Feature to file", "filename"_a, "feature"_a);
    docstring::FunctionDocInject(m_io, "write_feature",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_pose_graph",
            [](const std::string &filename) {
                py::gil_scoped_release release;
                pipelines::registration::PoseGraph pose_graph;
                ReadPoseGraph(filename, pose_graph);
                return pose_graph;
            },
            "Function to read PoseGraph from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pose_graph",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pose_graph",
            [](const std::string &filename,
               const pipelines::registration::PoseGraph pose_graph) {
                py::gil_scoped_release release;
                WritePoseGraph(filename, pose_graph);
            },
            "Function to write PoseGraph to file", "filename"_a,
            "pose_graph"_a);
    docstring::FunctionDocInject(m_io, "write_pose_graph",
                                 map_shared_argument_docstrings);

#ifdef BUILD_AZURE_KINECT
    m_io.def(
            "read_azure_kinect_sensor_config",
            [](const std::string &filename) {
                AzureKinectSensorConfig config;
                bool success = ReadIJsonConvertibleFromJSON(filename, config);
                if (!success) {
                    utility::LogWarning(
                            "Invalid sensor config {}, using default instead",
                            filename);
                    return AzureKinectSensorConfig();
                }
                return config;
            },
            "Function to read Azure Kinect sensor config from file",
            "filename"_a);
    docstring::FunctionDocInject(m_io, "read_azure_kinect_sensor_config",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_azure_kinect_sensor_config",
            [](const std::string &filename,
               const AzureKinectSensorConfig config) {
                return WriteIJsonConvertibleToJSON(filename, config);
            },
            "Function to write Azure Kinect sensor config to file",
            "filename"_a, "config"_a);
    docstring::FunctionDocInject(m_io, "write_azure_kinect_sensor_config",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_azure_kinect_mkv_metadata",
            [](const std::string &filename) {
                MKVMetadata metadata;
                bool success = ReadIJsonConvertibleFromJSON(filename, metadata);
                if (!success) {
                    utility::LogWarning(
                            "Invalid mkv metadata {}, using default instead",
                            filename);
                    return MKVMetadata();
                }
                return metadata;
            },
            "Function to read Azure Kinect metadata from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_azure_kinect_mkv_metadata",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_azure_kinect_mkv_metadata",
            [](const std::string &filename, const MKVMetadata metadata) {
                return WriteIJsonConvertibleToJSON(filename, metadata);
            },
            "Function to write Azure Kinect metadata to file", "filename"_a,
            "config"_a);
    docstring::FunctionDocInject(m_io, "write_azure_kinect_mkv_metadata",
                                 map_shared_argument_docstrings);
#endif
}

}  // namespace io
}  // namespace open3d
