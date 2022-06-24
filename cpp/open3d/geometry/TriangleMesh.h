// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <Eigen/Core>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open3d/geometry/Image.h"
#include "open3d/geometry/MeshBase.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace geometry {

class PointCloud;
class TetraMesh;

/// \class TriangleMesh
///
/// \brief Triangle mesh contains vertices and triangles represented by the
/// indices to the vertices.
///
/// Optionally, the mesh may also contain triangle normals, vertex normals and
/// vertex colors.
class TriangleMesh : public MeshBase {
public:
    /// \brief Default Constructor.
    TriangleMesh() : MeshBase(Geometry::GeometryType::TriangleMesh) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param vertices list of vertices.
    /// \param triangles list of triangles.
    TriangleMesh(const std::vector<Eigen::Vector3d> &vertices,
                 const std::vector<Eigen::Vector3i> &triangles)
        : MeshBase(Geometry::GeometryType::TriangleMesh, vertices),
          triangles_(triangles) {}
    ~TriangleMesh() override {}

public:
    virtual TriangleMesh &Clear() override;
    virtual TriangleMesh &Transform(
            const Eigen::Matrix4d &transformation) override;
    virtual TriangleMesh &Rotate(const Eigen::Matrix3d &R,
                                 const Eigen::Vector3d &center) override;

public:
    TriangleMesh &operator+=(const TriangleMesh &mesh);
    TriangleMesh operator+(const TriangleMesh &mesh) const;

    /// Returns `true` if the mesh contains triangles.
    bool HasTriangles() const {
        return vertices_.size() > 0 && triangles_.size() > 0;
    }

    /// Returns `true` if the mesh contains triangle normals.
    bool HasTriangleNormals() const {
        return HasTriangles() && triangles_.size() == triangle_normals_.size();
    }

    /// Returns `true` if the mesh contains adjacency normals.
    bool HasAdjacencyList() const {
        return vertices_.size() > 0 &&
               adjacency_list_.size() == vertices_.size();
    }

    bool HasTriangleUvs() const {
        return HasTriangles() && triangle_uvs_.size() == 3 * triangles_.size();
    }

    /// Returns `true` if the mesh has texture.
    bool HasTextures() const {
        bool is_all_texture_valid = std::accumulate(
                textures_.begin(), textures_.end(), true,
                [](bool a, const Image &b) { return a && !b.IsEmpty(); });
        return !textures_.empty() && is_all_texture_valid;
    }

    bool HasMaterials() const { return !materials_.empty(); }

    bool HasTriangleMaterialIds() const {
        return HasTriangles() &&
               triangle_material_ids_.size() == triangles_.size();
    }

    /// Normalize both triangle normals and vertex normals to length 1.
    TriangleMesh &NormalizeNormals() {
        MeshBase::NormalizeNormals();
        for (size_t i = 0; i < triangle_normals_.size(); i++) {
            triangle_normals_[i].normalize();
            if (std::isnan(triangle_normals_[i](0))) {
                triangle_normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
            }
        }
        return *this;
    }

    /// \brief Function to compute triangle normals, usually called before
    /// rendering.
    TriangleMesh &ComputeTriangleNormals(bool normalized = true);

    /// \brief Function to compute vertex normals, usually called before
    /// rendering.
    TriangleMesh &ComputeVertexNormals(bool normalized = true);

    /// \brief Function to compute adjacency list, call before adjacency list is
    /// needed.
    TriangleMesh &ComputeAdjacencyList();

    /// \brief Function that removes duplicated verties, i.e., vertices that
    /// have identical coordinates.
    TriangleMesh &RemoveDuplicatedVertices();

    /// \brief Function that removes duplicated triangles, i.e., removes
    /// triangles that reference the same three vertices and have the same
    /// orientation.
    TriangleMesh &RemoveDuplicatedTriangles();

    /// \brief This function removes vertices from the triangle mesh that are
    /// not referenced in any triangle of the mesh.
    TriangleMesh &RemoveUnreferencedVertices();

    /// \brief Function that removes degenerate triangles, i.e., triangles that
    /// reference a single vertex multiple times in a single triangle.
    ///
    /// They are usually the product of removing duplicated vertices.
    TriangleMesh &RemoveDegenerateTriangles();

    /// \brief Function that removes all non-manifold edges, by successively
    /// deleting triangles with the smallest surface area adjacent to the
    /// non-manifold edge until the number of adjacent triangles to the edge is
    /// `<= 2`.
    TriangleMesh &RemoveNonManifoldEdges();

    /// \brief Function that will merge close by vertices to a single one.
    /// The vertex position, normal and color will be the average of the
    /// vertices.
    ///
    /// \param eps defines the maximum distance of close by vertices.
    /// This function might help to close triangle soups.
    TriangleMesh &MergeCloseVertices(double eps);

    /// \brief Function to sharpen triangle mesh.
    ///
    /// The output value (\f$v_o\f$) is the input value (\f$v_i\f$) plus
    /// strength times the input value minus the sum of he adjacent values.
    /// \f$v_o = v_i + strength (v_i * |N| - \sum_{n \in N} v_n)\f$.
    ///
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    /// \param strength - The strength of the filter.
    std::shared_ptr<TriangleMesh> FilterSharpen(
            int number_of_iterations,
            double strength,
            FilterScope scope = FilterScope::All) const;

    /// \brief Function to smooth triangle mesh with simple neighbour average.
    ///
    /// \f$v_o = \frac{v_i + \sum_{n \in N} v_n)}{|N| + 1}\f$, with \f$v_i\f$
    /// being the input value, \f$v_o\f$ the output value, and \f$N\f$ is the
    /// set of adjacent neighbours.
    ///
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    std::shared_ptr<TriangleMesh> FilterSmoothSimple(
            int number_of_iterations,
            FilterScope scope = FilterScope::All) const;

    /// \brief Function to smooth triangle mesh using Laplacian.
    ///
    /// \f$v_o = v_i \cdot \lambda (\sum_{n \in N} w_n v_n - v_i)\f$,
    /// with \f$v_i\f$ being the input value, \f$v_o\f$ the output value,
    /// \f$N\f$ is the set of adjacent neighbours, \f$w_n\f$ is the weighting of
    /// the neighbour based on the inverse distance (closer neighbours have
    /// higher weight),
    ///
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    /// \param lambda_filter is the smoothing parameter.
    std::shared_ptr<TriangleMesh> FilterSmoothLaplacian(
            int number_of_iterations,
            double lambda_filter,
            FilterScope scope = FilterScope::All) const;

    /// \brief Function to smooth triangle mesh using method of Taubin,
    /// "Curve and Surface Smoothing Without Shrinkage", 1995.
    /// Applies in each iteration two times FilterSmoothLaplacian, first
    /// with lambda_filter and second with mu as smoothing parameter.
    /// This method avoids shrinkage of the triangle mesh.
    ///
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    /// \param lambda_filter is the filter parameter
    /// \param mu is the filter parameter
    std::shared_ptr<TriangleMesh> FilterSmoothTaubin(
            int number_of_iterations,
            double lambda_filter = 0.5,
            double mu = -0.53,
            FilterScope scope = FilterScope::All) const;

    /// Function that computes the Euler-Poincaré characteristic, i.e.,
    /// V + F - E, where V is the number of vertices, F is the number
    /// of triangles, and E is the number of edges.
    int EulerPoincareCharacteristic() const;

    /// Function that returns the non-manifold edges of the triangle mesh.
    /// If \param allow_boundary_edges is set to false, then also boundary
    /// edges are returned.
    std::vector<Eigen::Vector2i> GetNonManifoldEdges(
            bool allow_boundary_edges = true) const;

    /// Function that checks if the given triangle mesh is edge-manifold.
    /// A mesh is edge­-manifold if each edge is bounding either one or two
    /// triangles. If allow_boundary_edges is set to false, then this function
    /// returns false if there exists boundary edges.
    bool IsEdgeManifold(bool allow_boundary_edges = true) const;

    /// Function that returns a list of non-manifold vertex indices.
    /// A vertex is manifold if its star is edge‐manifold and edge‐connected.
    /// (Two or more faces connected only by a vertex and not by an edge.)
    std::vector<int> GetNonManifoldVertices() const;

    /// Function that checks if all vertices in the triangle mesh are manifold.
    /// A vertex is manifold if its star is edge‐manifold and edge‐connected.
    /// (Two or more faces connected only by a vertex and not by an edge.)
    bool IsVertexManifold() const;

    /// Function that returns a list of triangles that are intersecting the
    /// mesh.
    std::vector<Eigen::Vector2i> GetSelfIntersectingTriangles() const;

    /// Function that tests if the triangle mesh is self-intersecting.
    /// Tests each triangle pair for intersection.
    bool IsSelfIntersecting() const;

    /// Function that tests if the bounding boxes of the triangle meshes are
    /// intersecting.
    bool IsBoundingBoxIntersecting(const TriangleMesh &other) const;

    /// Function that tests if the triangle mesh intersects another triangle
    /// mesh. Tests each triangle against each other triangle.
    bool IsIntersecting(const TriangleMesh &other) const;

    /// Function that tests if the given triangle mesh is orientable, i.e.
    /// the triangles can be oriented in such a way that all normals point
    /// towards the outside.
    bool IsOrientable() const;

    /// Function that tests if the given triangle mesh is watertight by
    /// checking if it is vertex manifold and edge-manifold with no boundary
    /// edges, but not self-intersecting.
    bool IsWatertight() const;

    /// If the mesh is orientable then this function rearranges the
    /// triangles such that all normals point towards the
    /// outside/inside.
    bool OrientTriangles();

    /// Function that returns a map from edges (vertex0, vertex1) to the
    /// triangle indices the given edge belongs to.
    std::unordered_map<Eigen::Vector2i,
                       std::vector<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
    GetEdgeToTrianglesMap() const;

    /// Function that returns a map from edges (vertex0, vertex1) to the
    /// vertex (vertex2) indices the given edge belongs to.
    std::unordered_map<Eigen::Vector2i,
                       std::vector<int>,
                       utility::hash_eigen<Eigen::Vector2i>>
    GetEdgeToVerticesMap() const;

    /// Function that computes the area of a mesh triangle
    static double ComputeTriangleArea(const Eigen::Vector3d &p0,
                                      const Eigen::Vector3d &p1,
                                      const Eigen::Vector3d &p2);

    /// Function that computes the area of a mesh triangle identified by the
    /// triangle index
    double GetTriangleArea(size_t triangle_idx) const;

    static inline Eigen::Vector3i GetOrderedTriangle(int vidx0,
                                                     int vidx1,
                                                     int vidx2) {
        if (vidx0 > vidx2) {
            std::swap(vidx0, vidx2);
        }
        if (vidx0 > vidx1) {
            std::swap(vidx0, vidx1);
        }
        if (vidx1 > vidx2) {
            std::swap(vidx1, vidx2);
        }
        return Eigen::Vector3i(vidx0, vidx1, vidx2);
    }

    /// Function that computes the surface area of the mesh, i.e. the sum of
    /// the individual triangle surfaces.
    double GetSurfaceArea() const;

    /// Function that computes the surface area of the mesh, i.e. the sum of
    /// the individual triangle surfaces.
    double GetSurfaceArea(std::vector<double> &triangle_areas) const;

    /// Function that computes the volume of the mesh, under the condition that
    /// it is watertight and orientable.
    /// See Zhang and Chen, "Efficient feature extraction for 2D/3D objects in
    /// mesh representation", 2001.
    double GetVolume() const;

    /// Function that computes the plane equation from the three points.
    /// If the three points are co-linear, then this function returns the
    /// invalid plane (0, 0, 0, 0).
    static Eigen::Vector4d ComputeTrianglePlane(const Eigen::Vector3d &p0,
                                                const Eigen::Vector3d &p1,
                                                const Eigen::Vector3d &p2);

    /// Function that computes the plane equation of a mesh triangle identified
    /// by the triangle index.
    Eigen::Vector4d GetTrianglePlane(size_t triangle_idx) const;

    /// Helper function to get an edge with ordered vertex indices.
    static inline Eigen::Vector2i GetOrderedEdge(int vidx0, int vidx1) {
        return Eigen::Vector2i(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
    }

    /// Function to sample \param number_of_points points uniformly from the
    /// mesh.
    std::shared_ptr<PointCloud> SamplePointsUniformlyImpl(
            size_t number_of_points,
            std::vector<double> &triangle_areas,
            double surface_area,
            bool use_triangle_normal);

    /// Function to sample points uniformly from the mesh.
    ///
    /// \param number_of_points points uniformly from the mesh.
    /// \param use_triangle_normal Set to true to assign the triangle normals to
    /// the returned points instead of the interpolated vertex normals. The
    /// triangle normals will be computed and added to the mesh if necessary.
    std::shared_ptr<PointCloud> SamplePointsUniformly(
            size_t number_of_points, bool use_triangle_normal = false);

    /// Function to sample points from the mesh with Possion disk, based on the
    /// method presented in Yuksel, "Sample Elimination for Generating Poisson
    /// Disk Sample Sets", EUROGRAPHICS.
    ///
    /// \param number_of_points Number of points that should be sampled.
    /// \param init_factor Factor for the initial uniformly sampled PointCloud.
    /// This init PointCloud is used for sample elimination.
    /// \param pcl_init Initial PointCloud that is used for sample elimination.
    /// If this parameter is provided the init_factor is ignored.
    /// \param use_triangle_normal If True assigns the triangle normals instead
    /// of the interpolated vertex normals to the returned points. The triangle
    /// normals will be computed and added to the mesh if necessary.
    std::shared_ptr<PointCloud> SamplePointsPoissonDisk(
            size_t number_of_points,
            double init_factor = 5,
            const std::shared_ptr<PointCloud> pcl_init = nullptr,
            bool use_triangle_normal = false);

    /// Function to subdivide triangle mesh using the simple midpoint algorithm.
    /// Each triangle is subdivided into four triangles per iteration and the
    /// new vertices lie on the midpoint of the triangle edges.
    /// \param number_of_iterations defines a single iteration splits each
    /// triangle into four triangles that cover the same surface.
    std::shared_ptr<TriangleMesh> SubdivideMidpoint(
            int number_of_iterations) const;

    /// Function to subdivide triangle mesh using Loop's scheme.
    /// Cf. Charles T. Loop, "Smooth subdivision surfaces based on triangles",
    /// 1987. Each triangle is subdivided into four triangles per iteration.
    /// \param number_of_iterations defines a single iteration splits each
    /// triangle into four triangles that cover the same surface.
    std::shared_ptr<TriangleMesh> SubdivideLoop(int number_of_iterations) const;

    /// Function to simplify mesh using Vertex Clustering.
    /// The result can be a non-manifold mesh.
    /// \param voxel_size - The size of the voxel within vertices are pooled.
    /// \param contraction - Method to aggregate vertex information. Average
    /// computes a simple average, Quadric minimizes the distance to the
    /// adjacent planes.
    std::shared_ptr<TriangleMesh> SimplifyVertexClustering(
            double voxel_size,
            SimplificationContraction contraction =
                    SimplificationContraction::Average) const;

    /// Function to simplify mesh using Quadric Error Metric Decimation by
    /// Garland and Heckbert.
    /// \param target_number_of_triangles defines the number of triangles that
    /// the simplified mesh should have. It is not guaranteed that this number
    /// will be reached.
    /// \param maximum_error defines the maximum error where a vertex is allowed
    /// to be merged
    /// \param boundary_weight a weight applied to edge vertices used to
    /// preserve boundaries
    std::shared_ptr<TriangleMesh> SimplifyQuadricDecimation(
            int target_number_of_triangles,
            double maximum_error,
            double boundary_weight) const;

    /// Function to select points from \p input TriangleMesh into
    /// output TriangleMesh
    /// Vertices with indices in \p indices are selected.
    /// \param indices defines Indices of vertices to be selected.
    /// \param cleanup If true it automatically calls
    /// TriangleMesh::RemoveDuplicatedVertices,
    /// TriangleMesh::RemoveDuplicatedTriangles,
    /// TriangleMesh::RemoveUnreferencedVertices, and
    /// TriangleMesh::RemoveDegenerateTriangles
    std::shared_ptr<TriangleMesh> SelectByIndex(
            const std::vector<size_t> &indices, bool cleanup = true) const;

    /// Function to crop pointcloud into output pointcloud
    /// All points with coordinates outside the bounding box \p bbox are
    /// clipped.
    /// \param bbox defines the input Axis Aligned Bounding Box.
    std::shared_ptr<TriangleMesh> Crop(
            const AxisAlignedBoundingBox &bbox) const;

    /// Function to crop pointcloud into output pointcloud
    /// All points with coordinates outside the bounding box \p bbox are
    /// clipped.
    /// \param bbox defines the input Oriented Bounding Box.
    std::shared_ptr<TriangleMesh> Crop(const OrientedBoundingBox &bbox) const;

    /// \brief Function that clusters connected triangles, i.e., triangles that
    /// are connected via edges are assigned the same cluster index.
    ///
    /// \return A vector that contains the cluster index per
    /// triangle, a second vector contains the number of triangles per
    /// cluster, and a third vector contains the surface area per cluster.
    std::tuple<std::vector<int>, std::vector<size_t>, std::vector<double>>
    ClusterConnectedTriangles() const;

    /// \brief This function removes the triangles with index in
    /// \p triangle_indices. Call \ref RemoveUnreferencedVertices to clean up
    /// vertices afterwards.
    ///
    /// \param triangle_indices Indices of the triangles that should be
    /// removed.
    void RemoveTrianglesByIndex(const std::vector<size_t> &triangle_indices);

    /// \brief This function removes the triangles that are masked in
    /// \p triangle_mask. Call \ref RemoveUnreferencedVertices to clean up
    /// vertices afterwards.
    ///
    /// \param triangle_mask Mask of triangles that should be removed.
    /// Should have same size as \ref triangles_.
    void RemoveTrianglesByMask(const std::vector<bool> &triangle_mask);

    /// \brief This function removes the vertices with index in
    /// \p vertex_indices. Note that also all triangles associated with the
    /// vertices are removeds.
    ///
    /// \param vertex_indices Indices of the vertices that should be
    /// removed.
    void RemoveVerticesByIndex(const std::vector<size_t> &vertex_indices);

    /// \brief This function removes the vertices that are masked in
    /// \p vertex_mask. Note that also all triangles associated with the
    /// vertices are removed.
    ///
    /// \param vertex_mask Mask of vertices that should be removed.
    /// Should have same size as \ref vertices_.
    void RemoveVerticesByMask(const std::vector<bool> &vertex_mask);

    /// \brief This function deforms the mesh using the method by
    /// Sorkine and Alexa, "As-Rigid-As-Possible Surface Modeling", 2007.
    ///
    /// \param constraint_vertex_indices Indices of the triangle vertices that
    /// should be constrained by the vertex positions in
    /// constraint_vertex_positions.
    /// \param constraint_vertex_positions Vertex positions used for the
    /// constraints.
    /// \param max_iter maximum number of iterations to minimize energy
    /// functional.
    /// \param energy energy model that should be optimized
    /// \param smoothed_alpha alpha parameter of the smoothed ARAP model
    /// \return The deformed TriangleMesh
    std::shared_ptr<TriangleMesh> DeformAsRigidAsPossible(
            const std::vector<int> &constraint_vertex_indices,
            const std::vector<Eigen::Vector3d> &constraint_vertex_positions,
            size_t max_iter,
            DeformAsRigidAsPossibleEnergy energy =
                    DeformAsRigidAsPossibleEnergy::Spokes,
            double smoothed_alpha = 0.01) const;

    /// \brief Alpha shapes are a generalization of the convex hull. With
    /// decreasing alpha value the shape schrinks and creates cavities.
    /// See Edelsbrunner and Muecke, "Three-Dimensional Alpha Shapes", 1994.
    /// \param pcd PointCloud for what the alpha shape should be computed.
    /// \param alpha parameter to control the shape. A very big value will
    /// give a shape close to the convex hull.
    /// \param tetra_mesh If not a nullptr, then uses this to construct the
    /// alpha shape. Otherwise, ComputeDelaunayTetrahedralization is called.
    /// \param pt_map Optional map from tetra_mesh vertex indices to pcd
    /// points.
    /// \return TriangleMesh of the alpha shape.
    static std::shared_ptr<TriangleMesh> CreateFromPointCloudAlphaShape(
            const PointCloud &pcd,
            double alpha,
            std::shared_ptr<TetraMesh> tetra_mesh = nullptr,
            std::vector<size_t> *pt_map = nullptr);

    /// Function that computes a triangle mesh from an oriented PointCloud \p
    /// pcd. This implements the Ball Pivoting algorithm proposed in F.
    /// Bernardini et al., "The ball-pivoting algorithm for surface
    /// reconstruction", 1999. The implementation is also based on the
    /// algorithms outlined in Digne, "An Analysis and Implementation of a
    /// Parallel Ball Pivoting Algorithm", 2014. The surface reconstruction is
    /// done by rolling a ball with a given radius (cf. \p radii) over the
    /// point cloud, whenever the ball touches three points a triangle is
    /// created.
    /// \param pcd defines the PointCloud from which the TriangleMesh surface is
    /// reconstructed. Has to contain normals.
    /// \param radii defines the radii of
    /// the ball that are used for the surface reconstruction.
    static std::shared_ptr<TriangleMesh> CreateFromPointCloudBallPivoting(
            const PointCloud &pcd, const std::vector<double> &radii);

    /// \brief Function that computes a triangle mesh from an oriented
    /// PointCloud pcd. This implements the Screened Poisson Reconstruction
    /// proposed in Kazhdan and Hoppe, "Screened Poisson Surface
    /// Reconstruction", 2013. This function uses the original implementation by
    /// Kazhdan. See https://github.com/mkazhdan/PoissonRecon
    ///
    /// \param pcd PointCloud with normals and optionally colors.
    /// \param depth Maximum depth of the tree that will be used for surface
    /// reconstruction. Running at depth d corresponds to solving on a grid
    /// whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the
    /// reconstructor adapts the octree to the sampling density, the specified
    /// reconstruction depth is only an upper bound.
    /// \param width Specifies the target width of the finest level octree
    /// cells. This parameter is ignored if depth is specified.
    /// \param scale Specifies the ratio between the diameter of the cube used
    /// for reconstruction and the diameter of the samples' bounding cube.
    /// \param linear_fit If true, the reconstructor use linear interpolation to
    /// estimate the positions of iso-vertices.
    /// \param n_threads Number of threads used for reconstruction. Set to -1 to
    /// automatically determine it.
    /// \return The estimated TriangleMesh, and per vertex density values that
    /// can be used to to trim the mesh.
    static std::tuple<std::shared_ptr<TriangleMesh>, std::vector<double>>
    CreateFromPointCloudPoisson(const PointCloud &pcd,
                                size_t depth = 8,
                                float width = 0.0f,
                                float scale = 1.1f,
                                bool linear_fit = false,
                                int n_threads = -1);

    /// Factory function to create a tetrahedron mesh (trianglemeshfactory.cpp).
    /// the mesh centroid will be at (0,0,0) and \p radius defines the
    /// distance from the center to the mesh vertices.
    /// \param radius defines the distance from centroid to mesh vetices.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<TriangleMesh> CreateTetrahedron(
            double radius = 1.0, bool create_uv_map = false);

    /// Factory function to create an octahedron mesh (trianglemeshfactory.cpp).
    /// the mesh centroid will be at (0,0,0) and \p radius defines the
    /// distance from the center to the mesh vertices.
    /// \param radius defines the distance from centroid to mesh vetices.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<TriangleMesh> CreateOctahedron(
            double radius = 1.0, bool create_uv_map = false);

    /// Factory function to create an icosahedron mesh
    /// (trianglemeshfactory.cpp). The mesh centroid will be at (0,0,0) and
    /// \param radius defines the distance from the center to the mesh vertices.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<TriangleMesh> CreateIcosahedron(
            double radius = 1.0, bool create_uv_map = false);

    /// Factory function to create a box mesh (TriangleMeshFactory.cpp)
    /// The left bottom corner on the front will be placed at (0, 0, 0).
    /// \param width is x-directional length.
    /// \param height is y-directional length.
    /// \param depth is z-directional length.
    /// \param create_uv_map add default UV map to the shape.
    /// \param map_texture_to_each_face if true, maps the entire texture image
    /// to each face. If false, sets the default uv map to the mesh.
    static std::shared_ptr<TriangleMesh> CreateBox(
            double width = 1.0,
            double height = 1.0,
            double depth = 1.0,
            bool create_uv_map = false,
            bool map_texture_to_each_face = false);

    /// Factory function to create a sphere mesh (TriangleMeshFactory.cpp)
    /// The sphere with radius will be centered at (0, 0, 0).
    /// Its axis is aligned with z-axis.
    /// \param radius defines radius of the sphere.
    /// \param resolution defines the resolution of the sphere. The longitudes
    /// will be split into resolution segments (i.e. there are resolution + 1
    /// latitude lines including the north and south pole). The latitudes will
    /// be split into `2 * resolution segments (i.e. there are 2 * resolution
    /// longitude lines.)
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<TriangleMesh> CreateSphere(
            double radius = 1.0,
            int resolution = 20,
            bool create_uv_map = false);

    /// Factory function to create a cylinder mesh (TriangleMeshFactory.cpp)
    /// The axis of the cylinder will be from (0, 0, -height/2) to (0, 0,
    /// height/2). The circle with radius will be split into
    /// resolution segments. The height will be split into split
    /// segments.
    /// \param radius defines the radius of the cylinder.
    /// \param height defines the height of the cylinder.
    /// \param resolution defines that the circle will be split into resolution
    /// segments.
    /// \param split defines that the height will be split into split segments.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<TriangleMesh> CreateCylinder(
            double radius = 1.0,
            double height = 2.0,
            int resolution = 20,
            int split = 4,
            bool create_uv_map = false);

    /// Factory function to create a cone mesh (TriangleMeshFactory.cpp)
    /// The axis of the cone will be from (0, 0, 0) to (0, 0, height).
    /// The circle with radius will be split into resolution
    /// segments. The height will be split into split segments.
    /// \param radius defines the radius of the cone.
    /// \param height defines the height of the cone.
    /// \param resolution defines that the circle will be split into resolution
    /// segments.
    /// \param split defines that the height will be split into split segments.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<TriangleMesh> CreateCone(double radius = 1.0,
                                                    double height = 2.0,
                                                    int resolution = 20,
                                                    int split = 1,
                                                    bool create_uv_map = false);

    /// Factory function to create a torus mesh (TriangleMeshFactory.cpp)
    /// The torus will be centered at (0, 0, 0) and a radius of
    /// torus_radius. The tube of the torus will have a radius of
    /// tube_radius. The number of segments in radial and tubular direction are
    /// radial_resolution and tubular_resolution respectively.
    ///
    /// \param torus_radius defines the radius from the center of the torus to
    /// the center of the tube.
    /// \param tube_radius defines the radius of the torus tube.
    /// \param radial_resolution defines the he number of segments along the
    /// radial direction.
    /// \param tubular_resolution defines the number of segments along the
    /// tubular direction.
    static std::shared_ptr<TriangleMesh> CreateTorus(
            double torus_radius = 1.0,
            double tube_radius = 0.5,
            int radial_resolution = 30,
            int tubular_resolution = 20);

    /// Factory function to create an arrow mesh (TriangleMeshFactory.cpp)
    /// The axis of the cone with cone_radius will be along the z-axis.
    /// The cylinder with cylinder_radius is from
    /// (0, 0, 0) to (0, 0, cylinder_height), and
    /// the cone is from (0, 0, cylinder_height)
    /// to (0, 0, cylinder_height + cone_height).
    /// The cone will be split into resolution segments.
    /// The cylinder_height will be split into cylinder_split
    /// segments. The cone_height will be split into cone_split
    /// segments.
    //
    /// \param cylinder_radius defines the radius of the cylinder.
    /// \param cone_radius defines the radius of the cone.
    /// \param cylinder_height defines the height of the cylinder. The cylinder
    /// is from (0, 0, 0) to (0, 0, cylinder_height)
    /// \param cone_height defines the height of the cone. The axis of the cone
    /// will be from (0, 0, cylinder_height) to (0, 0, cylinder_height +
    /// cone_height).
    /// \param resolution defines the cone will be split into resolution
    /// segments.
    /// \param cylinder_split defines the cylinder_height will be split into
    /// cylinder_split segments.
    /// \param cone_split defines the cone_height will be split into cone_split
    /// segments.
    static std::shared_ptr<TriangleMesh> CreateArrow(
            double cylinder_radius = 1.0,
            double cone_radius = 1.5,
            double cylinder_height = 5.0,
            double cone_height = 4.0,
            int resolution = 20,
            int cylinder_split = 4,
            int cone_split = 1);

    /// Factory function to create a coordinate frame mesh
    /// (TriangleMeshFactory.cpp).
    /// arrows respectively. \p size is the length of the axes.
    /// \param size defines the size of the coordinate frame.
    /// \param origin defines the origin of the coordinate frame.
    static std::shared_ptr<TriangleMesh> CreateCoordinateFrame(
            double size = 1.0,
            const Eigen::Vector3d &origin = Eigen::Vector3d(0.0, 0.0, 0.0));

    /// Factory function to create a Mobius strip.
    /// \param length_split defines the number of segments along the Mobius
    /// strip.
    /// \param width_split defines the number of segments along the width
    /// of the Mobius strip.\param twists defines the number of twists of the
    /// strip.
    /// \param radius defines the radius of the Mobius strip.
    /// \param flatness controls the height of the strip.
    /// \param width controls the width of the Mobius strip.
    /// \param scale is used to scale the entire Mobius strip.
    static std::shared_ptr<TriangleMesh> CreateMobius(int length_split = 70,
                                                      int width_split = 15,
                                                      int twists = 1,
                                                      double radius = 1,
                                                      double flatness = 1,
                                                      double width = 1,
                                                      double scale = 1);

protected:
    // Forward child class type to avoid indirect nonvirtual base
    TriangleMesh(Geometry::GeometryType type) : MeshBase(type) {}

    void FilterSmoothLaplacianHelper(
            std::shared_ptr<TriangleMesh> &mesh,
            const std::vector<Eigen::Vector3d> &prev_vertices,
            const std::vector<Eigen::Vector3d> &prev_vertex_normals,
            const std::vector<Eigen::Vector3d> &prev_vertex_colors,
            const std::vector<std::unordered_set<int>> &adjacency_list,
            double lambda_filter,
            bool filter_vertex,
            bool filter_normal,
            bool filter_color) const;

    /// \brief Function that computes for each edge in the triangle mesh and
    /// passed as parameter edges_to_vertices the cot weight.
    ///
    /// \param edges_to_vertices map from edge to vector of neighbouring
    /// vertices.
    /// \param min_weight minimum weight returned. Weights smaller than this
    /// get clamped.
    /// \return cot weight per edge.
    std::unordered_map<Eigen::Vector2i,
                       double,
                       utility::hash_eigen<Eigen::Vector2i>>
    ComputeEdgeWeightsCot(
            const std::unordered_map<Eigen::Vector2i,
                                     std::vector<int>,
                                     utility::hash_eigen<Eigen::Vector2i>>
                    &edges_to_vertices,
            double min_weight = std::numeric_limits<double>::lowest()) const;

public:
    /// List of triangles denoted by the index of points forming the triangle.
    std::vector<Eigen::Vector3i> triangles_;
    /// Triangle normals.
    std::vector<Eigen::Vector3d> triangle_normals_;
    /// The set adjacency_list[i] contains the indices of adjacent vertices of
    /// vertex i.
    std::vector<std::unordered_set<int>> adjacency_list_;
    /// List of uv coordinates per triangle.
    std::vector<Eigen::Vector2d> triangle_uvs_;

    struct Material {
        struct MaterialParameter {
            float f4[4] = {0};

            MaterialParameter() {
                f4[0] = 0;
                f4[1] = 0;
                f4[2] = 0;
                f4[3] = 0;
            }

            MaterialParameter(const float v1,
                              const float v2,
                              const float v3,
                              const float v4) {
                f4[0] = v1;
                f4[1] = v2;
                f4[2] = v3;
                f4[3] = v4;
            }

            MaterialParameter(const float v1, const float v2, const float v3) {
                f4[0] = v1;
                f4[1] = v2;
                f4[2] = v3;
                f4[3] = 1;
            }

            MaterialParameter(const float v1, const float v2) {
                f4[0] = v1;
                f4[1] = v2;
                f4[2] = 0;
                f4[3] = 0;
            }

            explicit MaterialParameter(const float v1) {
                f4[0] = v1;
                f4[1] = 0;
                f4[2] = 0;
                f4[3] = 0;
            }

            static MaterialParameter CreateRGB(const float r,
                                               const float g,
                                               const float b) {
                return {r, g, b, 1.f};
            }

            float r() const { return f4[0]; }
            float g() const { return f4[1]; }
            float b() const { return f4[2]; }
            float a() const { return f4[3]; }
        };

        MaterialParameter baseColor;
        float baseMetallic = 0.f;
        float baseRoughness = 1.f;
        float baseReflectance = 0.5f;
        float baseClearCoat = 0.f;
        float baseClearCoatRoughness = 0.f;
        float baseAnisotropy = 0.f;

        std::shared_ptr<Image> albedo;
        std::shared_ptr<Image> normalMap;
        std::shared_ptr<Image> ambientOcclusion;
        std::shared_ptr<Image> metallic;
        std::shared_ptr<Image> roughness;
        std::shared_ptr<Image> reflectance;
        std::shared_ptr<Image> clearCoat;
        std::shared_ptr<Image> clearCoatRoughness;
        std::shared_ptr<Image> anisotropy;

        std::unordered_map<std::string, MaterialParameter> floatParameters;
        std::unordered_map<std::string, Image> additionalMaps;
    };

    std::unordered_map<std::string, Material> materials_;

    /// List of material ids.
    std::vector<int> triangle_material_ids_;
    /// Textures of the image.
    std::vector<Image> textures_;
};

}  // namespace geometry
}  // namespace open3d
