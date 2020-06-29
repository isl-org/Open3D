// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#pragma once

#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace pipelines {
namespace mesh_factory {

/// Factory function to create a tetrahedron mesh (trianglemeshfactory.cpp).
/// the mesh centroid will be at (0,0,0) and \param radius defines the
/// distance from the center to the mesh vertices.
/// \param radius defines the distance from centroid to mesh vetices.
std::shared_ptr<geometry::TriangleMesh> CreateTetrahedron(double radius = 1.0);

/// Factory function to create an octahedron mesh (trianglemeshfactory.cpp).
/// the mesh centroid will be at (0,0,0) and \param radius defines the
/// distance from the center to the mesh vertices.
/// \param radius defines the distance from centroid to mesh vetices.
std::shared_ptr<geometry::TriangleMesh> CreateOctahedron(double radius = 1.0);

/// Factory function to create an icosahedron mesh
/// (trianglemeshfactory.cpp). The mesh centroid will be at (0,0,0) and
/// \param radius defines the distance from the center to the mesh vertices.
std::shared_ptr<geometry::TriangleMesh> CreateIcosahedron(double radius = 1.0);

/// Factory function to create a box mesh
/// The left bottom corner on the front will be placed at (0, 0, 0).
/// \param width is x-directional length.
/// \param height is y-directional length.
/// \param depth is z-directional length.
std::shared_ptr<geometry::TriangleMesh> CreateBox(double width = 1.0,
                                                  double height = 1.0,
                                                  double depth = 1.0);

/// Factory function to create a sphere mesh
/// The sphere with radius will be centered at (0, 0, 0).
/// Its axis is aligned with z-axis.
/// \param radius defines radius of the sphere.
/// \param resolution defines the resolution of the sphere. The longitudes
/// will be split into resolution segments (i.e. there are resolution + 1
/// latitude lines including the north and south pole). The latitudes will
/// be split into `2 * resolution segments (i.e. there are 2 * resolution
/// longitude lines.)
std::shared_ptr<geometry::TriangleMesh> CreateSphere(double radius = 1.0,
                                                     int resolution = 20);

/// Factory function to create a cylinder mesh
/// The axis of the cylinder will be from (0, 0, -height/2) to (0, 0,
/// height/2). The circle with radius will be split into
/// resolution segments. The height will be split into split
/// segments.
/// \param radius defines the radius of the cylinder.
/// \param height defines the height of the cylinder.
/// \param resolution defines that the circle will be split into resolution
/// segments \param split defines that the height will be split into split
/// segments.
std::shared_ptr<geometry::TriangleMesh> CreateCylinder(double radius = 1.0,
                                                       double height = 2.0,
                                                       int resolution = 20,
                                                       int split = 4);

/// Factory function to create a cone mesh
/// The axis of the cone will be from (0, 0, 0) to (0, 0, height).
/// The circle with radius will be split into resolution
/// segments. The height will be split into split segments.
/// \param radius defines the radius of the cone.
/// \param height defines the height of the cone.
/// \param resolution defines that the circle will be split into resolution
/// segments \param split defines that the height will be split into split
/// segments.
std::shared_ptr<geometry::TriangleMesh> CreateCone(double radius = 1.0,
                                                   double height = 2.0,
                                                   int resolution = 20,
                                                   int split = 1);

/// Factory function to create a torus mesh
/// The torus will be centered at (0, 0, 0) and a radius of
/// torus_radius. The tube of the torus will have a radius of
/// tube_radius. The number of segments in radial and tubular direction are
/// radial_resolution and tubular_resolution respectively.
/// \param torus_radius defines the radius from the center of the torus to
/// the center of the tube. \param tube_radius defines the radius of the
/// torus tube. \param radial_resolution defines the he number of segments
/// along the radial direction. \param tubular_resolution defines the number
/// of segments along the tubular direction.
std::shared_ptr<geometry::TriangleMesh> CreateTorus(
        double torus_radius = 1.0,
        double tube_radius = 0.5,
        int radial_resolution = 30,
        int tubular_resolution = 20);

/// Factory function to create an arrow mesh
/// The axis of the cone with cone_radius will be along the z-axis.
/// The cylinder with cylinder_radius is from
/// (0, 0, 0) to (0, 0, cylinder_height), and
/// the cone is from (0, 0, cylinder_height)
/// to (0, 0, cylinder_height + cone_height).
/// The cone will be split into resolution segments.
/// The cylinder_height will be split into cylinder_split
/// segments. The cone_height will be split into cone_split
/// segments.
/// \param cylinder_radius defines the radius of the cylinder.
/// \param cone_radius defines the radius of the cone.
/// \param cylinder_height defines the height of the cylinder. The cylinder
/// is from (0, 0, 0) to (0, 0, cylinder_height) \param cone_height defines
/// the height of the cone. The axis of the cone will be from (0, 0,
/// cylinder_height) to (0, 0, cylinder_height + cone_height). \param
/// resolution defines the cone will be split into resolution segments.
/// \param cylinder_split defines the cylinder_height will be split into
/// cylinder_split segments. \param cone_split defines the cone_height will
/// be split into cone_split segments.
std::shared_ptr<geometry::TriangleMesh> CreateArrow(
        double cylinder_radius = 1.0,
        double cone_radius = 1.5,
        double cylinder_height = 5.0,
        double cone_height = 4.0,
        int resolution = 20,
        int cylinder_split = 4,
        int cone_split = 1);

/// Factory function to create a coordinate frame mesh
/// arrows respectively. \param size is the length of the axes.
/// \param size defines the size of the coordinate frame.
/// \param origin defines the origin of the coordinate frame.
std::shared_ptr<geometry::TriangleMesh> CreateCoordinateFrame(
        double size = 1.0,
        const Eigen::Vector3d &origin = Eigen::Vector3d(0.0, 0.0, 0.0));

/// Factory function to create a Moebius strip.
/// \param length_split defines the number of segments along the Moebius
/// strip. \param width_split defines the number of segments along the width
/// of the Moebius strip. \param twists defines the number of twists of the
/// strip.
/// \param radius defines the radius of the Moebius strip.
/// \param flatness controls the height of the strip.
/// \param width controls the width of the Moebius strip.
/// \param scale is used to scale the entire Moebius strip.
std::shared_ptr<geometry::TriangleMesh> CreateMoebius(int length_split = 70,
                                                      int width_split = 15,
                                                      int twists = 1,
                                                      double radius = 1,
                                                      double flatness = 1,
                                                      double width = 1,
                                                      double scale = 1);
}  // namespace mesh_factory
}  // namespace pipelines
}  // namespace open3d
