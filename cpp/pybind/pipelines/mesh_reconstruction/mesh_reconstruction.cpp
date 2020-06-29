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

#include "open3d/pipelines/mesh_reconstruction/TriangleMeshReconstruction.h"
#include "pybind/docstring.h"

namespace open3d {

void pybind_mesh_reconstruction(py::module &m) {
    m.def("create_from_point_cloud_alpha_shape",
          [](const geometry::PointCloud &pcd, double alpha) {
              return pipelines::mesh_reconstruction::ReconstructAlphaShape(
                      pcd, alpha);
          },
          "Alpha shapes are a generalization of the convex hull. "
          "With decreasing alpha value the shape schrinks and "
          "creates cavities. See Edelsbrunner and Muecke, "
          "\"Three-Dimensional Alpha Shapes\", 1994.",
          "pcd"_a, "alpha"_a);
    m.def("create_from_point_cloud_alpha_shape",
          &pipelines::mesh_reconstruction::ReconstructAlphaShape,
          "Alpha shapes are a generalization of the convex hull. "
          "With decreasing alpha value the shape schrinks and "
          "creates cavities. See Edelsbrunner and Muecke, "
          "\"Three-Dimensional Alpha Shapes\", 1994.",
          "pcd"_a, "alpha"_a, "tetra_mesh"_a, "pt_map"_a);
    m.def("create_from_point_cloud_ball_pivoting",
          &pipelines::mesh_reconstruction::ReconstructBallPivoting,
          "Function that computes a triangle mesh from a oriented "
          "PointCloud. This implements the Ball Pivoting algorithm "
          "proposed in F. Bernardini et al., \"The ball-pivoting "
          "algorithm for surface reconstruction\", 1999. The "
          "implementation is also based on the algorithms outlined "
          "in Digne, \"An Analysis and Implementation of a Parallel "
          "Ball Pivoting Algorithm\", 2014. The surface "
          "reconstruction is done by rolling a ball with a given "
          "radius over the point cloud, whenever the ball touches "
          "three points a triangle is created.",
          "pcd"_a, "radii"_a);
    m.def("create_from_point_cloud_poisson",
          &pipelines::mesh_reconstruction::ReconstructPoisson,
          "Function that computes a triangle mesh from a "
          "oriented PointCloud pcd. This implements the Screened "
          "Poisson Reconstruction proposed in Kazhdan and Hoppe, "
          "\"Screened Poisson Surface Reconstruction\", 2013. "
          "This function uses the original implementation by "
          "Kazhdan. See https://github.com/mkazhdan/PoissonRecon",
          "pcd"_a, "depth"_a = 8, "width"_a = 0, "scale"_a = 1.1,
          "linear_fit"_a = false);

    docstring::FunctionDocInject(
            m, "create_from_point_cloud_alpha_shape",
            {{"pcd",
              "PointCloud from whicht the TriangleMesh surface is "
              "reconstructed."},
             {"alpha",
              "Parameter to control the shape. A very big value will give a "
              "shape close to the convex hull."},
             {"tetra_mesh",
              "If not None, than uses this to construct the alpha shape. "
              "Otherwise, TetraMesh is computed from pcd."},
             {"pt_map",
              "Optional map from tetra_mesh vertex indices to pcd points."}});
    docstring::FunctionDocInject(
            m, "create_from_point_cloud_ball_pivoting",
            {{"pcd",
              "PointCloud from which the TriangleMesh surface is "
              "reconstructed. Has to contain normals."},
             {"radii",
              "The radii of the ball that are used for the surface "
              "reconstruction."}});
    docstring::FunctionDocInject(
            m, "create_from_point_cloud_poisson",
            {{"pcd",
              "PointCloud from which the TriangleMesh surface is "
              "reconstructed. Has to contain normals."},
             {"depth",
              "Maximum depth of the tree that will be used for surface "
              "reconstruction. Running at depth d corresponds to solving on a "
              "grid whose resolution is no larger than 2^d x 2^d x 2^d. Note "
              "that since the reconstructor adapts the octree to the sampling "
              "density, the specified reconstruction depth is only an upper "
              "bound."},
             {"width",
              "Specifies the target width of the finest level octree cells. "
              "This parameter is ignored if depth is specified"},
             {"scale",
              "Specifies the ratio between the diameter of the cube used for "
              "reconstruction and the diameter of the samples' bounding cube."},
             {"linear_fit",
              "If true, the reconstructor will use linear interpolation to "
              "estimate the positions of iso-vertices."}});
}

}  // namespace open3d
