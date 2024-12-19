// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>
#include <unordered_map>

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include "pybind/docstring.h"
#include "pybind/t/io/io.h"

namespace open3d {
namespace t {
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
                 "present on the mesh."},
                {"write_vertex_colors",
                 "Set to ``False`` to not write any vertex colors, even if "
                 "present on the mesh."},
                {"write_triangle_uvs",
                 "Set to ``False`` to not write any triangle uvs, even if "
                 "present on the mesh. For ``obj`` format, mtl file is saved "
                 "only when ``True`` is set."},
                // Entities
                {"config", "AzureKinectSensor's config file."},
                {"pointcloud", "The ``PointCloud`` object for I/O."},
                {"mesh", "The ``TriangleMesh`` object for I/O."},
                {"line_set", "The ``LineSet`` object for I/O."},
                {"image", "The ``Image`` object for I/O."},
                {"voxel_grid", "The ``VoxelGrid`` object for I/O."},
                {"trajectory",
                 "The ``PinholeCameraTrajectory`` object for I/O."},
                {"intrinsic", "The ``PinholeCameraIntrinsic`` object for I/O."},
                {"parameters",
                 "The ``PinholeCameraParameters`` object for I/O."},
                {"pose_graph", "The ``PoseGraph`` object for I/O."},
                {"feature", "The ``Feature`` object for I/O."},
                {"print_progress",
                 "If set to true a progress bar is visualized in the console."},
};

void pybind_class_io_declarations(py::module &m_io) {
    py::class_<DepthNoiseSimulator> depth_noise_simulator(
            m_io, "DepthNoiseSimulator",
            R"(Simulate depth image noise from a given noise distortion model. The distortion model is based on *Teichman et. al. "Unsupervised intrinsic calibration of depth sensors via SLAM" RSS 2009*. Also see <http://redwood-data.org/indoor/dataset.html>__

Example::

    import open3d as o3d

    # Redwood Indoor LivingRoom1 (Augmented ICL-NUIM)
    # http://redwood-data.org/indoor/
    data = o3d.data.RedwoodIndoorLivingRoom1()
    noise_model_path = data.noise_model_path
    im_src_path = data.depth_paths[0]
    depth_scale = 1000.0

    # Read clean depth image (uint16)
    im_src = o3d.t.io.read_image(im_src_path)

    # Run noise model simulation
    simulator = o3d.t.io.DepthNoiseSimulator(noise_model_path)
    im_dst = simulator.simulate(im_src, depth_scale=depth_scale)

    # Save noisy depth image (uint16)
    o3d.t.io.write_image("noisy_depth.png", im_dst)
            )");
}

void pybind_class_io_definitions(py::module &m_io) {
    // open3d::t::geometry::Image
    m_io.def(
            "read_image",
            [](const fs::path &filename) {
                py::gil_scoped_release release;
                geometry::Image image;
                ReadImage(filename.string(), image);
                return image;
            },
            "Function to read image from file.", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_image",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_image",
            [](const fs::path &filename, const geometry::Image &image,
               int quality) {
                py::gil_scoped_release release;
                return WriteImage(filename.string(), image, quality);
            },
            "Function to write Image to file.", "filename"_a, "image"_a,
            "quality"_a = kOpen3DImageIODefaultQuality);
    docstring::FunctionDocInject(m_io, "write_image",
                                 map_shared_argument_docstrings);

    // open3d::t::geometry::PointCloud
    m_io.def(
            "read_point_cloud",
            [](const fs::path &filename, const std::string &format,
               bool remove_nan_points, bool remove_infinite_points,
               bool print_progress) {
                py::gil_scoped_release release;
                t::geometry::PointCloud pcd;
                ReadPointCloud(filename.string(), pcd,
                               {format, remove_nan_points,
                                remove_infinite_points, print_progress});
                return pcd;
            },
            "Function to read PointCloud with tensor attributes from file.",
            "filename"_a, "format"_a = "auto", "remove_nan_points"_a = false,
            "remove_infinite_points"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_point_cloud",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_point_cloud",
            [](const fs::path &filename,
               const t::geometry::PointCloud &pointcloud, bool write_ascii,
               bool compressed, bool print_progress) {
                py::gil_scoped_release release;
                return WritePointCloud(
                        filename.string(), pointcloud,
                        {write_ascii, compressed, print_progress});
            },
            "Function to write PointCloud with tensor attributes to file.",
            "filename"_a, "pointcloud"_a, "write_ascii"_a = false,
            "compressed"_a = false, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_point_cloud",
                                 map_shared_argument_docstrings);

    // open3d::geometry::TriangleMesh
    m_io.def(
            "read_triangle_mesh",
            [](const fs::path &filename, bool enable_post_processing,
               bool print_progress) {
                py::gil_scoped_release release;
                t::geometry::TriangleMesh mesh;
                open3d::io::ReadTriangleMeshOptions opt;
                opt.enable_post_processing = enable_post_processing;
                opt.print_progress = print_progress;
                ReadTriangleMesh(filename.string(), mesh, opt);
                return mesh;
            },
            "Function to read TriangleMesh from file", "filename"_a,
            "enable_post_processing"_a = false, "print_progress"_a = false,
            R"doc(The general entrance for reading a TriangleMesh from a file.
The function calls read functions based on the extension name of filename.
Supported formats are `obj, ply, stl, off, gltf, glb, fbx`.

The following example reads a triangle mesh with the .ply extension::
    import open3d as o3d
    mesh = o3d.t.io.read_triangle_mesh('mesh.ply')

Args:
    filename (str): Path to the mesh file.
    enable_post_processing (bool): If True enables post-processing.
        Post-processing will
          - triangulate meshes with polygonal faces
          - remove redundant materials
          - pretransform vertices
          - generate face normals if needed

        For more information see ASSIMPs documentation on the flags
        `aiProcessPreset_TargetRealtime_Fast, aiProcess_RemoveRedundantMaterials,
        aiProcess_OptimizeMeshes, aiProcess_PreTransformVertices`.

        Note that identical vertices will always be joined regardless of whether
        post-processing is enabled or not, which changes the number of vertices
        in the mesh.

        The `ply`-format is not affected by the post-processing.

    print_progress (bool): If True print the reading progress to the terminal.

Returns:
    Returns the mesh object. On failure an empty mesh is returned.
)doc");

    m_io.def(
            "write_triangle_mesh",
            [](const fs::path &filename, const t::geometry::TriangleMesh &mesh,
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

    // DepthNoiseSimulator
    auto depth_noise_simulator = static_cast<py::class_<DepthNoiseSimulator>>(
            m_io.attr("DepthNoiseSimulator"));
    depth_noise_simulator.def(py::init([](const fs::path &fielname) {
                                  return DepthNoiseSimulator(fielname.string());
                              }),
                              "noise_model_path"_a);
    depth_noise_simulator.def("simulate", &DepthNoiseSimulator::Simulate,
                              "im_src"_a, "depth_scale"_a = 1000.0f,
                              "Apply noise model to a depth image.");
    depth_noise_simulator.def(
            "enable_deterministic_debug_mode",
            &DepthNoiseSimulator::EnableDeterministicDebugMode,
            "Enable deterministic debug mode. All normally distributed noise "
            "will be replaced by 0.");
    depth_noise_simulator.def_property_readonly(
            "noise_model", &DepthNoiseSimulator::GetNoiseModel,
            "The noise model tensor.");
    docstring::ClassMethodDocInject(
            m_io, "DepthNoiseSimulator", "__init__",
            {{"noise_model_path",
              "Path to the noise model file. See "
              "http://redwood-data.org/indoor/dataset.html for the format. Or, "
              "you may use one of our example datasets, e.g., "
              "RedwoodIndoorLivingRoom1."}});
    docstring::ClassMethodDocInject(
            m_io, "DepthNoiseSimulator", "simulate",
            {{"im_src",
              "Source depth image, must be with dtype UInt16 or Float32, "
              "channels==1."},
             {"depth_scale",
              "Scale factor to the depth image. As a sanity check, if the "
              "dtype is Float32, the depth_scale must be 1.0. If the dtype is "
              "is UInt16, the depth_scale is typically larger than 1.0, e.g. "
              "it can be 1000.0."}});
    docstring::ClassMethodDocInject(m_io, "DepthNoiseSimulator",
                                    "enable_deterministic_debug_mode");
}

}  // namespace io
}  // namespace t
}  // namespace open3d
