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

#pragma once

#include <Eigen/Core>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/MeshBase.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {
namespace geometry {

class PointCloud;
class TetraMesh;

class TriangleMesh : public MeshBase {
public:
    TriangleMesh() : MeshBase(Geometry::GeometryType::TriangleMesh) {}
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
                                 bool center = true) override;

public:
    TriangleMesh &operator+=(const TriangleMesh &mesh);
    TriangleMesh operator+(const TriangleMesh &mesh) const;

    bool HasTriangles() const {
        return vertices_.size() > 0 && triangles_.size() > 0;
    }

    bool HasTriangleNormals() const {
        return HasTriangles() && triangles_.size() == triangle_normals_.size();
    }

    bool HasAdjacencyList() const {
        return vertices_.size() > 0 &&
               adjacency_list_.size() == vertices_.size();
    }

    bool HasTriangleUvs() const {
        return HasTriangles() && triangle_uvs_.size() == 3 * triangles_.size();
    }

    bool HasTexture() const {
        bool is_all_texture_valid = std::accumulate(
                textures_.begin(), textures_.end(), true,
                [](bool a, const Image &b) { return a && !b.IsEmpty(); });
        return !textures_.empty() && is_all_texture_valid;
    }

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

    /// Function to compute triangle normals, usually called before rendering
    TriangleMesh &ComputeTriangleNormals(bool normalized = true);

    /// Function to compute vertex normals, usually called before rendering
    TriangleMesh &ComputeVertexNormals(bool normalized = true);

    /// Function to compute adjacency list, call before adjacency list is needed
    TriangleMesh &ComputeAdjacencyList();

    /// Function that removes duplicated verties, i.e., vertices that have
    /// identical coordinates.
    TriangleMesh &RemoveDuplicatedVertices();

    /// Function that removes duplicated triangles, i.e., removes triangles
    /// that reference the same three vertices, independent of their order.
    TriangleMesh &RemoveDuplicatedTriangles();

    /// This function removes vertices from the triangle mesh that are not
    /// referenced in any triangle of the mesh.
    TriangleMesh &RemoveUnreferencedVertices();

    /// Function that removes degenerate triangles, i.e., triangles that
    /// reference a single vertex multiple times in a single triangle.
    /// They are usually the product of removing duplicated vertices.
    TriangleMesh &RemoveDegenerateTriangles();

    /// Function that removes all non-manifold edges, by successively deleting
    /// triangles with the smallest surface area adjacent to the non-manifold
    /// edge until the number of adjacent triangles to the edge is `<= 2`.
    TriangleMesh &RemoveNonManifoldEdges();

    /// Function that will merge close by vertices to a single one. The vertex
    /// position, normal and color will be the average of the vertices. The
    /// parameter \param eps defines the maximum distance of close by vertices.
    /// This function might help to close triangle soups.
    TriangleMesh &MergeCloseVertices(double eps);

    /// Function to sharpen triangle mesh. The output value ($v_o$) is the
    /// input value ($v_i$) plus \param strength times the input value minus
    /// the sum of he adjacent values.
    /// $v_o = v_i x strength (v_i * |N| - \sum_{n \in N} v_n)$.
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    std::shared_ptr<TriangleMesh> FilterSharpen(
            int number_of_iterations,
            double strength,
            FilterScope scope = FilterScope::All) const;

    /// Function to smooth triangle mesh with simple neighbour average.
    /// $v_o = \frac{v_i + \sum_{n \in N} v_n)}{|N| + 1}$, with $v_i$
    /// being the input value, $v_o$ the output value, and $N$ is the
    /// set of adjacent neighbours.
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    std::shared_ptr<TriangleMesh> FilterSmoothSimple(
            int number_of_iterations,
            FilterScope scope = FilterScope::All) const;

    /// Function to smooth triangle mesh using Laplacian.
    /// $v_o = v_i \cdot \lambda (sum_{n \in N} w_n v_n - v_i)$,
    /// with $v_i$ being the input value, $v_o$ the output value, $N$ is the
    /// set of adjacent neighbours, $w_n$ is the weighting of the neighbour
    /// based on the inverse distance (closer neighbours have higher weight),
    /// and \param lambda is the smoothing parameter.
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    std::shared_ptr<TriangleMesh> FilterSmoothLaplacian(
            int number_of_iterations,
            double lambda,
            FilterScope scope = FilterScope::All) const;

    /// Function to smooth triangle mesh using method of Taubin,
    /// "Curve and Surface Smoothing Without Shrinkage", 1995.
    /// Applies in each iteration two times FilterSmoothLaplacian, first
    /// with \param lambda and second with \param mu as smoothing parameter.
    /// This method avoids shrinkage of the triangle mesh.
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    std::shared_ptr<TriangleMesh> FilterSmoothTaubin(
            int number_of_iterations,
            double lambda = 0.5,
            double mu = -0.53,
            FilterScope scope = FilterScope::All) const;

    /// Function that computes the Euler-Poincaré characteristic, i.e.,
    /// V + F - E, where V is the number of vertices, F is the number
    /// of triangles, and E is the number of edges.
    int EulerPoincareCharacteristic() const;

    /// Function that returns the non-manifold edges of the triangle mesh.
    /// If \param allow_boundary_edges is set to false, than also boundary
    /// edges are returned
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
    /// the triangles can oriented in such a way that all normals point
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
                       utility::hash_eigen::hash<Eigen::Vector2i>>
    GetEdgeToTrianglesMap() const;

    /// Function that returns a map from edges (vertex0, vertex1) to the
    /// vertex (vertex2) indices the given edge belongs to.
    std::unordered_map<Eigen::Vector2i,
                       std::vector<int>,
                       utility::hash_eigen::hash<Eigen::Vector2i>>
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
    /// mesh
    std::shared_ptr<PointCloud> SamplePointsUniformlyImpl(
            size_t number_of_points,
            std::vector<double> &triangle_areas,
            double surface_area) const;

    /// Function to sample \param number_of_points points uniformly from the
    /// mesh
    std::shared_ptr<PointCloud> SamplePointsUniformly(
            size_t number_of_points) const;

    /// Function to sample \param number_of_points points (blue noise).
    /// Based on the method presented in Yuksel, "Sample Elimination for
    /// Generating Poisson Disk Sample Sets", EUROGRAPHICS, 2015 The PointCloud
    /// \param pcl_init is used for sample elimination if given, otherwise a
    /// PointCloud is first uniformly sampled with \param init_number_of_points
    /// x \param number_of_points number of points.
    std::shared_ptr<PointCloud> SamplePointsPoissonDisk(
            size_t number_of_points,
            double init_factor = 5,
            const std::shared_ptr<PointCloud> pcl_init = nullptr) const;

    /// Function to subdivide triangle mesh using the simple midpoint algorithm.
    /// Each triangle is subdivided into four triangles per iteration and the
    /// new vertices lie on the midpoint of the triangle edges.
    std::shared_ptr<TriangleMesh> SubdivideMidpoint(
            int number_of_iterations) const;

    /// Function to subdivide triangle mesh using Loop's scheme.
    /// Cf. Charles T. Loop, "Smooth subdivision surfaces based on triangles",
    /// 1987. Each triangle is subdivided into four triangles per iteration.
    std::shared_ptr<TriangleMesh> SubdivideLoop(int number_of_iterations) const;

    /// Function to simplify mesh using Vertex Clustering.
    /// The result can be a non-manifold mesh.
    std::shared_ptr<TriangleMesh> SimplifyVertexClustering(
            double voxel_size,
            SimplificationContraction contraction =
                    SimplificationContraction::Average) const;

    /// Function to simplify mesh using Quadric Error Metric Decimation by
    /// Garland and Heckbert.
    std::shared_ptr<TriangleMesh> SimplifyQuadricDecimation(
            int target_number_of_triangles) const;

    /// Function to select points from \param input TriangleMesh into
    /// \return output TriangleMesh
    /// Vertices with indices in \param indices are selected.
    std::shared_ptr<TriangleMesh> SelectDownSample(
            const std::vector<size_t> &indices) const;

    /// Function to crop pointcloud into output pointcloud
    /// All points with coordinates outside the bounding box \param bbox are
    /// clipped.
    std::shared_ptr<TriangleMesh> Crop(
            const AxisAlignedBoundingBox &bbox) const;

    /// Function to crop pointcloud into output pointcloud
    /// All points with coordinates outside the bounding box \param bbox are
    /// clipped.
    std::shared_ptr<TriangleMesh> Crop(const OrientedBoundingBox &bbox) const;

    /// /brief Function that clusters connected triangles, i.e., triangles that
    /// are connected via edges are assigned the same cluster index.
    ///
    /// \return a vector that contains the cluster index per
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
    /// \param triangle_indices Indices of the triangles that should be
    /// removed.
    void RemoveVerticesByIndex(const std::vector<size_t> &vertex_indices);

    /// \brief This function removes the vertices that are masked in
    /// \p vertex_mask. Note that also all triangles associated with the
    /// vertices are removed..
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
    /// functional. \return The deformed TriangleMesh
    std::shared_ptr<TriangleMesh> DeformAsRigidAsPossible(
            const std::vector<int> &constraint_vertex_indices,
            const std::vector<Eigen::Vector3d> &constraint_vertex_positions,
            size_t max_iter) const;

    /// \brief Alpha shapes are a generalization of the convex hull. With
    /// decreasing alpha value the shape schrinks and creates cavities.
    /// See Edelsbrunner and Muecke, "Three-Dimensional Alpha Shapes", 1994.
    /// \param pcd PointCloud for what the alpha shape should be computed.
    /// \param alpha parameter to controll the shape. A very big value will
    /// give a shape close to the convex hull.
    /// \param tetra_mesh If not a nullptr, than uses this to construct the
    /// alpha shape. Otherwise, ComputeDelaunayTetrahedralization is called.
    /// \param pt_map Optional map from tetra_mesh vertex indices to pcd
    /// points.
    /// \return TriangleMesh of the alpha shape.
    static std::shared_ptr<TriangleMesh> CreateFromPointCloudAlphaShape(
            const PointCloud &pcd,
            double alpha,
            std::shared_ptr<TetraMesh> tetra_mesh = nullptr,
            std::vector<size_t> *pt_map = nullptr);

    /// Function that computes a triangle mesh from a oriented PointCloud \param
    /// pcd. This implements the Ball Pivoting algorithm proposed in F.
    /// Bernardini et al., "The ball-pivoting algorithm for surface
    /// reconstruction", 1999. The implementation is also based on the
    /// algorithms outlined in Digne, "An Analysis and Implementation of a
    /// Parallel Ball Pivoting Algorithm", 2014. The surface reconstruction is
    /// done by rolling a ball with a given radius (cf. \param radii) over the
    /// point cloud, whenever the ball touches three points a triangle is
    /// created.
    static std::shared_ptr<TriangleMesh> CreateFromPointCloudBallPivoting(
            const PointCloud &pcd, const std::vector<double> &radii);

    /// \brief Function that computes a triangle mesh from a oriented PointCloud
    /// pcd. This implements the Screened Poisson Reconstruction proposed in
    /// Kazhdan and Hoppe, "Screened Poisson Surface Reconstruction", 2013.
    /// This function uses the original implementation by Kazhdan. See
    /// https://github.com/mkazhdan/PoissonRecon
    ///
    /// \param pcd PointCloud with normals and optionally colors.
    /// \param depth Maximum depth of the tree that will be used for surface
    /// reconstruction. Running at depth d corresponds to solving on a grid
    /// whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the
    /// reconstructor adapts the octree to the sampling density, the specified
    /// reconstruction depth is only an upper bound.
    /// \param width Specifies the
    /// target width of the finest level octree cells. This parameter is ignored
    /// if depth is specified.
    /// \param scale Specifies the ratio between the
    /// diameter of the cube used for reconstruction and the diameter of the
    /// samples' bounding cube. \param linear_fit If true, the reconstructor use
    /// linear interpolation to estimate the positions of iso-vertices.
    /// \return The estimated TriangleMesh, and per vertex densitie values that
    /// can be used to to trim the mesh.
    static std::tuple<std::shared_ptr<TriangleMesh>, std::vector<double>>
    CreateFromPointCloudPoisson(const PointCloud &pcd,
                                size_t depth = 8,
                                size_t width = 0,
                                float scale = 1.1f,
                                bool linear_fit = false);

    /// Factory function to create a tetrahedron mesh (trianglemeshfactory.cpp).
    /// the mesh centroid will be at (0,0,0) and \param radius defines the
    /// distance from the center to the mesh vertices.
    static std::shared_ptr<TriangleMesh> CreateTetrahedron(double radius = 1.0);

    /// Factory function to create an octahedron mesh (trianglemeshfactory.cpp).
    /// the mesh centroid will be at (0,0,0) and \param radius defines the
    /// distance from the center to the mesh vertices.
    static std::shared_ptr<TriangleMesh> CreateOctahedron(double radius = 1.0);

    /// Factory function to create an icosahedron mesh
    /// (trianglemeshfactory.cpp). the mesh centroid will be at (0,0,0) and
    /// \param radius defines the distance from the center to the mesh vertices.
    static std::shared_ptr<TriangleMesh> CreateIcosahedron(double radius = 1.0);

    /// Factory function to create a box mesh (TriangleMeshFactory.cpp)
    /// The left bottom corner on the front will be placed at (0, 0, 0).
    /// The \param width is x-directional length, and \param height and \param
    /// depth are y- and z-directional lengths respectively.
    static std::shared_ptr<TriangleMesh> CreateBox(double width = 1.0,
                                                   double height = 1.0,
                                                   double depth = 1.0);

    /// Factory function to create a sphere mesh (TriangleMeshFactory.cpp)
    /// The sphere with \param radius will be centered at (0, 0, 0).
    /// Its axis is aligned with z-axis.
    /// The longitudes will be split into \param resolution segments.
    /// The latitudes will be split into \param resolution * 2 segments.
    static std::shared_ptr<TriangleMesh> CreateSphere(double radius = 1.0,
                                                      int resolution = 20);

    /// Factory function to create a cylinder mesh (TriangleMeshFactory.cpp)
    /// The axis of the cylinder will be from (0, 0, -height/2) to (0, 0,
    /// height/2). The circle with \param radius will be split into \param
    /// resolution segments. The \param height will be split into \param split
    /// segments.
    static std::shared_ptr<TriangleMesh> CreateCylinder(double radius = 1.0,
                                                        double height = 2.0,
                                                        int resolution = 20,
                                                        int split = 4);

    /// Factory function to create a cone mesh (TriangleMeshFactory.cpp)
    /// The axis of the cone will be from (0, 0, 0) to (0, 0, \param height).
    /// The circle with \param radius will be split into \param resolution
    /// segments. The height will be split into \param split segments.
    static std::shared_ptr<TriangleMesh> CreateCone(double radius = 1.0,
                                                    double height = 2.0,
                                                    int resolution = 20,
                                                    int split = 1);

    /// Factory function to create a torus mesh (TriangleMeshFactory.cpp)
    /// The torus will be centered at (0, 0, 0) and a radius of \param
    /// torus_radius. The tube of the torus will have a radius of \param
    /// tube_radius. The number of segments in radial and tubular direction are
    /// \param radial_resolution and \param tubular_resolution respectively.
    static std::shared_ptr<TriangleMesh> CreateTorus(
            double torus_radius = 1.0,
            double tube_radius = 0.5,
            int radial_resolution = 30,
            int tubular_resolution = 20);

    /// Factory function to create an arrow mesh (TriangleMeshFactory.cpp)
    /// The axis of the cone with \param cone_radius will be along the z-axis.
    /// The cylinder with \param cylinder_radius is from
    /// (0, 0, 0) to (0, 0, cylinder_height), and
    /// the cone is from (0, 0, cylinder_height)
    /// to (0, 0, cylinder_height + cone_height).
    /// The cone will be split into \param resolution segments.
    /// The \param cylinder_height will be split into \param cylinder_split
    /// segments. The \param cone_height will be split into \param cone_split
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
    /// (TriangleMeshFactory.cpp) The coordinate frame will be centered at
    /// \param origin The x, y, z axis will be rendered as red, green, and blue
    /// arrows respectively. \param size is the length of the axes.
    static std::shared_ptr<TriangleMesh> CreateCoordinateFrame(
            double size = 1.0,
            const Eigen::Vector3d &origin = Eigen::Vector3d(0.0, 0.0, 0.0));

    /// Factory function to create a Moebius strip. \param length_split
    /// defines the number of segments along the Moebius strip, \param
    /// width_split defines the number of segments along the width of
    /// the Moebius strip, \param twists defines the number of twists of the
    /// strip, \param radius defines the radius of the Moebius strip,
    /// \param flatness controls the height of the strip, \param width
    /// controls the width of the Moebius strip and \param scale is used
    /// to scale the entire Moebius strip.
    static std::shared_ptr<TriangleMesh> CreateMoebius(int length_split = 70,
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
            double lambda,
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
                       utility::hash_eigen::hash<Eigen::Vector2i>>
    ComputeEdgeWeightsCot(
            const std::unordered_map<Eigen::Vector2i,
                                     std::vector<int>,
                                     utility::hash_eigen::hash<Eigen::Vector2i>>
                    &edges_to_vertices,
            double min_weight = std::numeric_limits<double>::lowest()) const;

public:
    std::vector<Eigen::Vector3i> triangles_;
    std::vector<Eigen::Vector3d> triangle_normals_;
    std::vector<std::unordered_set<int>> adjacency_list_;
    std::vector<Eigen::Vector2d> triangle_uvs_;
    std::vector<int> triangle_material_ids_;
    std::vector<Image> textures_;
};

}  // namespace geometry
}  // namespace open3d
