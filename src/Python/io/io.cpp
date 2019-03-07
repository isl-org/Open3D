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

#include "Python/io/io.h"
#include "Python/docstring.h"
#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include "Open3D/Camera/PinholeCameraTrajectory.h"
#include "Open3D/IO/ClassIO/FeatureIO.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/IO/ClassIO/LineSetIO.h"
#include "Open3D/IO/ClassIO/IJsonConvertibleIO.h"
#include "Open3D/IO/ClassIO/PinholeCameraTrajectoryIO.h"
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/IO/ClassIO/PoseGraphIO.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/IO/ClassIO/VoxelGridIO.h"

using namespace open3d;

void pybind_io(py::module &m) {
    // This submodule is currently blank
    py::module m_io = m.def_submodule("io");

    // open3d::geometry::Image
    m_io.def("read_image",
             [](const std::string &filename) {
                 geometry::Image image;
                 io::ReadImage(filename, image);
                 return image;
             },
             "Function to read geometry::Image from file", "filename"_a);
    m_io.def("write_image",
             [](const std::string &filename, const geometry::Image &image,
                int quality) {
                 return io::WriteImage(filename, image, quality);
             },
             "Function to write geometry::Image to file", "filename"_a,
             "image"_a, "quality"_a = 90);

    // open3d::geometry::LineSet
    m_io.def("read_line_set",
             [](const std::string &filename, const std::string &format) {
                 geometry::LineSet line_set;
                 io::ReadLineSet(filename, line_set, format);
                 return line_set;
             },
             "Function to read geometry::LineSet from file", "filename"_a,
             "format"_a = "auto");
    m_io.def("write_line_set",
             [](const std::string &filename, const geometry::LineSet &line_set,
                bool write_ascii, bool compressed) {
                 return io::WriteLineSet(filename, line_set, write_ascii,
                                         compressed);
             },
             "Function to write geometry::LineSet to file", "filename"_a,
             "line_set"_a, "write_ascii"_a = false, "compressed"_a = false);

    // open3d::geometry::PointCloud
    m_io.def("read_point_cloud",
             [](const std::string &filename, const std::string &format) {
                 geometry::PointCloud pcd;
                 io::ReadPointCloud(filename, pcd, format);
                 return pcd;
             },
             "Function to read geometry::PointCloud from file", "filename"_a,
             "format"_a = "auto");
    m_io.def("write_point_cloud",
             [](const std::string &filename,
                const geometry::PointCloud &pointcloud, bool write_ascii,
                bool compressed) {
                 return io::WritePointCloud(filename, pointcloud, write_ascii,
                                            compressed);
             },
             "Function to write geometry::PointCloud to file", "filename"_a,
             "pointcloud"_a, "write_ascii"_a = false, "compressed"_a = false);

    // open3d::geometry::TriangleMesh
    m_io.def("read_triangle_mesh",
             [](const std::string &filename) {
                 geometry::TriangleMesh mesh;
                 io::ReadTriangleMesh(filename, mesh);
                 return mesh;
             },
             "Function to read geometry::TriangleMesh from file", "filename"_a);
    m_io.def("write_triangle_mesh",
             [](const std::string &filename, const geometry::TriangleMesh &mesh,
                bool write_ascii, bool compressed) {
                 return io::WriteTriangleMesh(filename, mesh, write_ascii,
                                              compressed);
             },
             "Function to write geometry::TriangleMesh to file", "filename"_a,
             "mesh"_a, "write_ascii"_a = false, "compressed"_a = false);

    // open3d::geometry::VoxelGrid
    m_io.def("read_voxel_grid",
             [](const std::string &filename, const std::string &format) {
                 geometry::VoxelGrid voxel_grid;
                 io::ReadVoxelGrid(filename, voxel_grid, format);
                 return voxel_grid;
             },
             "Function to read geometry::VoxelGrid from file", "filename"_a,
             "format"_a = "auto");
    m_io.def("write_voxel_grid",
             [](const std::string &filename,
                const geometry::VoxelGrid &voxel_grid, bool write_ascii,
                bool compressed) {
                 return io::WriteVoxelGrid(filename, voxel_grid, write_ascii,
                                           compressed);
             },
             "Function to write geometry::VoxelGrid to file", "filename"_a,
             "voxel_grid"_a, "write_ascii"_a = false, "compressed"_a = false);

    // open3d::camera
    m_io.def("read_pinhole_camera_intrinsic",
             [](const std::string &filename) {
                 camera::PinholeCameraIntrinsic intrinsic;
                 io::ReadIJsonConvertible(filename, intrinsic);
                 return intrinsic;
             },
             "Function to read camera::PinholeCameraIntrinsic from file",
             "filename"_a);
    m_io.def("write_pinhole_camera_intrinsic",
             [](const std::string &filename,
                const camera::PinholeCameraIntrinsic &intrinsic) {
                 return io::WriteIJsonConvertible(filename, intrinsic);
             },
             "Function to write camera::PinholeCameraIntrinsic to file",
             "filename"_a, "intrinsic"_a);
    m_io.def("read_pinhole_camera_parameters",
             [](const std::string &filename) {
                 camera::PinholeCameraParameters parameters;
                 io::ReadIJsonConvertible(filename, parameters);
                 return parameters;
             },
             "Function to read camera::PinholeCameraParameters from file",
             "filename"_a);
    m_io.def("write_pinhole_camera_parameters",
             [](const std::string &filename,
                const camera::PinholeCameraParameters &parameters) {
                 return io::WriteIJsonConvertible(filename, parameters);
             },
             "Function to write camera::PinholeCameraParameters to file",
             "filename"_a, "parameters"_a);
    m_io.def("read_pinhole_camera_trajectory",
             [](const std::string &filename) {
                 camera::PinholeCameraTrajectory trajectory;
                 io::ReadPinholeCameraTrajectory(filename, trajectory);
                 return trajectory;
             },
             "Function to read camera::PinholeCameraTrajectory from file",
             "filename"_a);
    m_io.def("write_pinhole_camera_trajectory",
             [](const std::string &filename,
                const camera::PinholeCameraTrajectory &trajectory) {
                 return io::WritePinholeCameraTrajectory(filename, trajectory);
             },
             "Function to write camera::PinholeCameraTrajectory to file",
             "filename"_a, "trajectory"_a);

    // open3d::registration
    m_io.def("read_feature",
             [](const std::string &filename) {
                 registration::Feature feature;
                 io::ReadFeature(filename, feature);
                 return feature;
             },
             "Function to read registration::Feature from file", "filename"_a);
    function_doc_inject(m_io, "read_feature", {{"filename", "Path to file"}});
    m_io.def("write_feature",
             [](const std::string &filename,
                const registration::Feature &feature) {
                 return io::WriteFeature(filename, feature);
             },
             "Function to write registration::Feature to file", "filename"_a,
             "feature"_a);
    m_io.def("read_pose_graph",
             [](const std::string &filename) {
                 registration::PoseGraph pose_graph;
                 io::ReadPoseGraph(filename, pose_graph);
                 return pose_graph;
             },
             "Function to read registration::PoseGraph from file",
             "filename"_a);
    m_io.def("write_pose_graph",
             [](const std::string &filename,
                const registration::PoseGraph pose_graph) {
                 io::WritePoseGraph(filename, pose_graph);
             },
             "Function to write registration::PoseGraph to file", "filename"_a,
             "pose_graph"_a);
}
