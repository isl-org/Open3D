// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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
#include "open3d/io/ModelIO.h"
#include "open3d/io/OctreeIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/io/PoseGraphIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/io/VoxelGridIO.h"
#include "open3d/visualization/rendering/Model.h"
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
                {"octree", "The ``Octree`` object for I/O"},
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

void pybind_class_io_declarations(py::module &m_io) {
    py::native_enum<FileGeometry>(m_io, "FileGeometry", "enum.IntFlag",
                                  "Geometry types")
            .value("CONTENTS_UNKNOWN", FileGeometry::CONTENTS_UNKNOWN)
            .value("CONTAINS_POINTS", FileGeometry::CONTAINS_POINTS)
            .value("CONTAINS_LINES", FileGeometry::CONTAINS_LINES)
            .value("CONTAINS_TRIANGLES", FileGeometry::CONTAINS_TRIANGLES)
            .export_values()
            .finalize();
}
void pybind_class_io_definitions(py::module &m_io) {
    m_io.def(
            "read_file_geometry_type", &ReadFileGeometryType,
            "Returns the type of geometry of the file. This is a faster way of "
            "determining the file type than attempting to read the file as a "
            "point cloud, mesh, or line set in turn.");

    // open3d::geometry::Image
    m_io.def(
            "read_image",
            [](const fs::path &filename) {
                py::gil_scoped_release release;
                geometry::Image image;
                ReadImage(filename.string(), image);
                return image;
            },
            "Function to read Image from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_image",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_image",
            [](const fs::path &filename, const geometry::Image &image,
               int quality) {
                py::gil_scoped_release release;
                return WriteImage(filename.string(), image, quality);
            },
            "Function to write Image to file", "filename"_a, "image"_a,
            "quality"_a = kOpen3DImageIODefaultQuality);
    docstring::FunctionDocInject(m_io, "write_image",
                                 map_shared_argument_docstrings);

    // open3d::geometry::LineSet
    m_io.def(
            "read_line_set",
            [](const fs::path &filename, const std::string &format,
               bool print_progress) {
                py::gil_scoped_release release;
                geometry::LineSet line_set;
                ReadLineSet(filename.string(), line_set, format,
                            print_progress);
                return line_set;
            },
            "Function to read LineSet from file", "filename"_a,
            "format"_a = "auto", "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_line_set",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_line_set",
            [](const fs::path &filename, const geometry::LineSet &line_set,
               bool write_ascii, bool compressed, bool print_progress) {
                py::gil_scoped_release release;
                return WriteLineSet(filename.string(), line_set, write_ascii,
                                    compressed, print_progress);
            },
            "Function to write LineSet to file", "filename"_a, "line_set"_a,
            "write_ascii"_a = false, "compressed"_a = false,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_line_set",
                                 map_shared_argument_docstrings);

    // open3d::geometry::PointCloud
    m_io.def(
            "read_point_cloud",
            [](const fs::path &filename, const std::string &format,
               bool remove_nan_points, bool remove_infinite_points,
               bool print_progress) {
                py::gil_scoped_release release;
                geometry::PointCloud pcd;
                ReadPointCloud(filename.string(), pcd,
                               {format, remove_nan_points,
                                remove_infinite_points, print_progress});
                return pcd;
            },
            "Function to read PointCloud from file", "filename"_a,
            "format"_a = "auto", "remove_nan_points"_a = false,
            "remove_infinite_points"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_point_cloud",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_point_cloud_from_bytes",
            [](const py::bytes &bytes, const std::string &format,
               bool remove_nan_points, bool remove_infinite_points,
               bool print_progress) {
                const char *dataptr = PYBIND11_BYTES_AS_STRING(bytes.ptr());
                auto length = PYBIND11_BYTES_SIZE(bytes.ptr());
                auto buffer = new unsigned char[length];
                // copy before releasing GIL
                std::memcpy(buffer, dataptr, length);
                py::gil_scoped_release release;
                geometry::PointCloud pcd;
                ReadPointCloud(reinterpret_cast<const unsigned char *>(buffer),
                               length, pcd,
                               {format, remove_nan_points,
                                remove_infinite_points, print_progress});
                delete[] buffer;
                return pcd;
            },
            "Function to read PointCloud from memory", "bytes"_a,
            "format"_a = "auto", "remove_nan_points"_a = false,
            "remove_infinite_points"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_point_cloud_from_bytes",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_point_cloud",
            [](const fs::path &filename, const geometry::PointCloud &pointcloud,
               const std::string &format, bool write_ascii, bool compressed,
               bool print_progress) {
                py::gil_scoped_release release;
                return WritePointCloud(
                        filename.string(), pointcloud,
                        {format, write_ascii, compressed, print_progress});
            },
            "Function to write PointCloud to file", "filename"_a,
            "pointcloud"_a, "format"_a = "auto", "write_ascii"_a = false,
            "compressed"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_point_cloud",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_point_cloud_to_bytes",
            [](const geometry::PointCloud &pointcloud,
               const std::string &format, bool write_ascii, bool compressed,
               bool print_progress) {
                py::gil_scoped_release release;
                size_t len = 0;
                unsigned char *buffer = nullptr;
                bool wrote = WritePointCloud(
                        buffer, len, pointcloud,
                        {format, write_ascii, compressed, print_progress});
                py::gil_scoped_acquire acquire;
                if (!wrote) {
                    return py::bytes();
                }
                auto ret =
                        py::bytes(reinterpret_cast<const char *>(buffer), len);
                delete[] buffer;
                return ret;
            },
            "Function to write PointCloud to memory", "pointcloud"_a,
            "format"_a = "auto", "write_ascii"_a = false,
            "compressed"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_point_cloud_to_bytes",
                                 map_shared_argument_docstrings);

    // open3d::geometry::TriangleMesh
    m_io.def(
            "read_triangle_mesh",
            [](const fs::path &filename, bool enable_post_processing,
               bool print_progress) {
                py::gil_scoped_release release;
                geometry::TriangleMesh mesh;
                ReadTriangleMeshOptions opt;
                opt.enable_post_processing = enable_post_processing;
                opt.print_progress = print_progress;
                ReadTriangleMesh(filename.string(), mesh, opt);
                return mesh;
            },
            "Function to read TriangleMesh from file", "filename"_a,
            "enable_post_processing"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_triangle_mesh",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_triangle_mesh",
            [](const fs::path &filename, const geometry::TriangleMesh &mesh,
               bool write_ascii, bool compressed, bool write_vertex_normals,
               bool write_vertex_colors, bool write_triangle_uvs,
               bool print_progress) {
                py::gil_scoped_release release;
                return WriteTriangleMesh(filename.string(), mesh, write_ascii,
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

    // open3d::visualization::rendering::TriangleMeshModel (Model.h)
    m_io.def(
            "read_triangle_model",
            [](const fs::path &filename, bool print_progress) {
                py::gil_scoped_release release;
                visualization::rendering::TriangleMeshModel model;
                ReadTriangleModelOptions opt;
                opt.print_progress = print_progress;
                ReadTriangleModel(filename.string(), model, opt);
                return model;
            },
            "Function to read visualization.rendering.TriangleMeshModel from "
            "file",
            "filename"_a, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_triangle_model",
                                 map_shared_argument_docstrings);

    // open3d::geometry::VoxelGrid
    m_io.def(
            "read_voxel_grid",
            [](const fs::path &filename, const std::string &format,
               bool print_progress) {
                py::gil_scoped_release release;
                geometry::VoxelGrid voxel_grid;
                ReadVoxelGrid(filename.string(), voxel_grid, format);
                return voxel_grid;
            },
            "Function to read VoxelGrid from file", "filename"_a,
            "format"_a = "auto", "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_voxel_grid",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_voxel_grid",
            [](const fs::path &filename, const geometry::VoxelGrid &voxel_grid,
               bool write_ascii, bool compressed, bool print_progress) {
                py::gil_scoped_release release;
                return WriteVoxelGrid(filename.string(), voxel_grid,
                                      write_ascii, compressed, print_progress);
            },
            "Function to write VoxelGrid to file", "filename"_a, "voxel_grid"_a,
            "write_ascii"_a = false, "compressed"_a = false,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_voxel_grid",
                                 map_shared_argument_docstrings);

    // open3d::geometry::Octree
    m_io.def(
            "read_octree",
            [](const fs::path &filename, const std::string &format) {
                py::gil_scoped_release release;
                geometry::Octree octree;
                ReadOctree(filename.string(), octree, format);
                return octree;
            },
            "Function to read Octree from file", "filename"_a,
            "format"_a = "auto");
    docstring::FunctionDocInject(m_io, "read_octree",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_octree",
            [](const fs::path &filename, const geometry::Octree &octree) {
                py::gil_scoped_release release;
                return WriteOctree(filename.string(), octree);
            },
            "Function to write Octree to file", "filename"_a, "octree"_a);
    docstring::FunctionDocInject(m_io, "write_octree",
                                 map_shared_argument_docstrings);

    // open3d::camera
    m_io.def(
            "read_pinhole_camera_intrinsic",
            [](const fs::path &filename) {
                py::gil_scoped_release release;
                camera::PinholeCameraIntrinsic intrinsic;
                ReadIJsonConvertible(filename.string(), intrinsic);
                return intrinsic;
            },
            "Function to read PinholeCameraIntrinsic from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pinhole_camera_intrinsic",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pinhole_camera_intrinsic",
            [](const fs::path &filename,
               const camera::PinholeCameraIntrinsic &intrinsic) {
                py::gil_scoped_release release;
                return WriteIJsonConvertible(filename.string(), intrinsic);
            },
            "Function to write PinholeCameraIntrinsic to file", "filename"_a,
            "intrinsic"_a);
    docstring::FunctionDocInject(m_io, "write_pinhole_camera_intrinsic",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_pinhole_camera_parameters",
            [](const fs::path &filename) {
                py::gil_scoped_release release;
                camera::PinholeCameraParameters parameters;
                ReadIJsonConvertible(filename.string(), parameters);
                return parameters;
            },
            "Function to read PinholeCameraParameters from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pinhole_camera_parameters",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pinhole_camera_parameters",
            [](const fs::path &filename,
               const camera::PinholeCameraParameters &parameters) {
                py::gil_scoped_release release;
                return WriteIJsonConvertible(filename.string(), parameters);
            },
            "Function to write PinholeCameraParameters to file", "filename"_a,
            "parameters"_a);
    docstring::FunctionDocInject(m_io, "write_pinhole_camera_parameters",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_pinhole_camera_trajectory",
            [](const fs::path &filename) {
                py::gil_scoped_release release;
                camera::PinholeCameraTrajectory trajectory;
                ReadPinholeCameraTrajectory(filename.string(), trajectory);
                return trajectory;
            },
            "Function to read PinholeCameraTrajectory from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pinhole_camera_trajectory",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pinhole_camera_trajectory",
            [](const fs::path &filename,
               const camera::PinholeCameraTrajectory &trajectory) {
                py::gil_scoped_release release;
                return WritePinholeCameraTrajectory(filename.string(),
                                                    trajectory);
            },
            "Function to write PinholeCameraTrajectory to file", "filename"_a,
            "trajectory"_a);
    docstring::FunctionDocInject(m_io, "write_pinhole_camera_trajectory",
                                 map_shared_argument_docstrings);

    // open3d::registration
    m_io.def(
            "read_feature",
            [](const fs::path &filename) {
                py::gil_scoped_release release;
                pipelines::registration::Feature feature;
                ReadFeature(filename.string(), feature);
                return feature;
            },
            "Function to read registration.Feature from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_feature",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_feature",
            [](const fs::path &filename,
               const pipelines::registration::Feature &feature) {
                py::gil_scoped_release release;
                return WriteFeature(filename.string(), feature);
            },
            "Function to write Feature to file", "filename"_a, "feature"_a);
    docstring::FunctionDocInject(m_io, "write_feature",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_pose_graph",
            [](const fs::path &filename) {
                py::gil_scoped_release release;
                pipelines::registration::PoseGraph pose_graph;
                ReadPoseGraph(filename.string(), pose_graph);
                return pose_graph;
            },
            "Function to read PoseGraph from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pose_graph",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pose_graph",
            [](const fs::path &filename,
               const pipelines::registration::PoseGraph pose_graph) {
                py::gil_scoped_release release;
                WritePoseGraph(filename.string(), pose_graph);
            },
            "Function to write PoseGraph to file", "filename"_a,
            "pose_graph"_a);
    docstring::FunctionDocInject(m_io, "write_pose_graph",
                                 map_shared_argument_docstrings);

#ifdef BUILD_AZURE_KINECT
    m_io.def(
            "read_azure_kinect_sensor_config",
            [](const fs::path &filename) {
                AzureKinectSensorConfig config;
                bool success =
                        ReadIJsonConvertibleFromJSON(filename.string(), config);
                if (!success) {
                    utility::LogWarning(
                            "Invalid sensor config {}, using default instead",
                            filename.string());
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
            [](const fs::path &filename, const AzureKinectSensorConfig config) {
                return WriteIJsonConvertibleToJSON(filename.string(), config);
            },
            "Function to write Azure Kinect sensor config to file",
            "filename"_a, "config"_a);
    docstring::FunctionDocInject(m_io, "write_azure_kinect_sensor_config",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_azure_kinect_mkv_metadata",
            [](const fs::path &filename) {
                MKVMetadata metadata;
                bool success = ReadIJsonConvertibleFromJSON(filename.string(),
                                                            metadata);
                if (!success) {
                    utility::LogWarning(
                            "Invalid mkv metadata {}, using default instead",
                            filename.string());
                    return MKVMetadata();
                }
                return metadata;
            },
            "Function to read Azure Kinect metadata from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_azure_kinect_mkv_metadata",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_azure_kinect_mkv_metadata",
            [](const fs::path &filename, const MKVMetadata metadata) {
                return WriteIJsonConvertibleToJSON(filename.string(), metadata);
            },
            "Function to write Azure Kinect metadata to file", "filename"_a,
            "config"_a);
    docstring::FunctionDocInject(m_io, "write_azure_kinect_mkv_metadata",
                                 map_shared_argument_docstrings);
#endif
}

}  // namespace io
}  // namespace open3d
