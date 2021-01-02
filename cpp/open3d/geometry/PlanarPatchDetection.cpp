// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <Eigen/Dense>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <unordered_map>

#include "libqhullcpp/PointCoordinates.h"
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullVertex.h"

#include "open3d/geometry/PlanarPatch.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace geometry {

namespace {

/// \class BoundaryVolumeHierarchy
///
/// \brief Breadth-first octree data structure
///
/// BoundaryVolumeHierarchy is different than Octree because it partitions
/// space on-demand in a breadth-first fashion rather than all at once in
/// depth-first order. Instead of specifying max_depth, this BVH can stop
/// partitioning once a node has less than min_points associated with it.
/// These features make BoundaryVolumeHierarchy more amenable to efficiently
/// detecting planes in hierarchical subregions of the point cloud.
class BoundaryVolumeHierarchy {
public:
    static constexpr int DIMENSION = 3;
    static constexpr size_t NUM_CHILDREN = 8;

    /// \brief Constructor for the root node of the octree.
    ///
    /// \param point_cloud is the associated set of points being partitioned
    BoundaryVolumeHierarchy(const PointCloud* point_cloud,
                            size_t min_points = 1,
                            double min_size = 0.0)
        : point_cloud_(point_cloud),
          min_points_(min_points),
          min_size_(min_size),
          leaf_(true),
          level_(0),
          child_index_(0) {
        // set origin of root node and size of each child node (cubes)
        const Eigen::Vector3d min_bound = point_cloud->GetMinBound();
        const Eigen::Vector3d max_bound = point_cloud->GetMaxBound();
        center_ = (min_bound + max_bound) / 2;
        const Eigen::Vector3d half_sizes = center_ - min_bound;
        size_ = 2 * half_sizes.maxCoeff();

        // since this is the root, all the point cloud's indices are contained
        indices_ = std::vector<size_t>(point_cloud->points_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
    }

    /// \brief Partition a leaf node's points into NUM_CHILDREN subdivisions
    void partition() {
        // Nothing to do if already partitioned
        if (!leaf_) return;

        // size of each child
        const double child_size = size_ / 2;

        // Does this node have enough data to be able to partition further
        if (indices_.size() <= min_points_ || child_size < min_size_ ||
            indices_.size() < 2) {
            // std::cout << " ** partition failed (" << (indices_.size() <= min_points_)
            //     << " " << (child_size < min_size_) << " " << (indices_.size() < 2)
            //     << ") ** "<< std::flush;
            // std::cout << "(" << indices_.size() << "vs" << min_points_ << ") " << std::flush;
            return;
        }

        // split points and create children
        for (const size_t& pidx : indices_) {
            // calculate child index comparing position to child center
            const size_t cidx = calculateChildIndex(point_cloud_->points_[pidx]);
            // if child does not yet exist, create and initialize
            if (children_[cidx] == nullptr) {
                const Eigen::Vector3d child_center =
                        calculateChildCenter(cidx, child_size);
                children_[cidx].reset(new BoundaryVolumeHierarchy(
                        point_cloud_, level_ + 1, child_center, min_points_, min_size_, child_size, cidx));
                children_[cidx]->indices_.reserve(indices_.size());
            }
            children_[cidx]->indices_.push_back(pidx);
        }

        // now that I have children, I am no longer a leaf node
        leaf_ = false;
        // for space efficiency, get rid of my list of points, which was
        // redistributed to my children
        indices_.clear(); // TODO: clear and scan children or keep in memory?
    }

    std::vector<size_t> indices() const {
        if (leaf_) {
            return indices_;
        } else {
            std::vector<size_t> indices;
            for (const auto& child : children_) {
                if (child) {
                    const auto& child_indices = child->indices();
                    indices.insert(indices.end(), child_indices.begin(), child_indices.end());
                }
            }
            return indices;
        }
    }

public:
    const PointCloud* point_cloud_;
    std::array<std::shared_ptr<BoundaryVolumeHierarchy>, NUM_CHILDREN>
            children_;
    Eigen::Vector3d center_;
    size_t min_points_;
    double min_size_;
    double size_;
    bool leaf_;
    size_t level_;
    size_t child_index_;
    std::vector<size_t> indices_;

private:
    /// \brief Private constructor for creating children.
    ///
    /// \param point_cloud is the (original) set of points being partitioned
    /// \param level in tree that this child lives on
    /// \param center coordinate of this child node
    /// \param size of this child (same for all nodes at this level)
    BoundaryVolumeHierarchy(const PointCloud* point_cloud,
                            size_t level,
                            const Eigen::Vector3d& center,
                            size_t min_points,
                            double min_size,
                            double size,
                            size_t child_index)
        : point_cloud_(point_cloud),
          center_(center),
          min_points_(min_points),
          min_size_(min_size),
          size_(size),
          leaf_(true),
          level_(level),
          child_index_(child_index) {}

    /// \brief Calculate the center coordinate of a child node.
    ///
    /// For a root node with center_ == (0, 0, 0) and size_ == 2,
    /// child_size == 1 and:
    ///   child_index 0: center == (-0.5, -0.5, -0.5)
    ///   child_index 1: center == (-0.5, -0.5,  0.5)
    ///   child_index 2: center == (-0.5,  0.5, -0.5)
    ///   child_index 3: center == (-0.5,  0.5,  0.5)
    ///   child_index 4: center == ( 0.5, -0.5, -0.5)
    ///   child_index 5: center == ( 0.5, -0.5,  0.5)
    ///   child_index 6: center == ( 0.5,  0.5, -0.5)
    ///   child_index 7: center == ( 0.5,  0.5,  0.5)
    ///
    /// \param child_index indicates which child
    /// \param child_size of the child's bounding volume (cube)
    Eigen::Vector3d calculateChildCenter(size_t child_index,
                                         double child_size) const {
        Eigen::Vector3d center;
        for (size_t d = 0; d < DIMENSION; d++) {
            const int signal = (((child_index & (1 << (DIMENSION - d - 1))) >>
                                 (DIMENSION - d - 1))
                                << 1) -
                               1;
            center(d) = center_(d) + (child_size / 2.) * signal;
        }
        return center;
    }

    /// \brief Calculate child index given a position
    ///
    /// \param position of point to find child index of
    size_t calculateChildIndex(const Eigen::Vector3d& position) const {
        size_t child_index = 0;
        for (size_t d = 0; d < DIMENSION; d++) {
            child_index |= (position(d) > center_(d)) << (DIMENSION - d - 1);
        }
        return child_index;
    }
};

/// \brief Calculate the median of a buffer of data
///
/// \param buffer Container of scalar data to find median of. A copy is made
/// so that buffer may be sorted.
/// \return Median of buffer data.
double getMedian(std::vector<double> buffer) {
    const size_t N = buffer.size();
    std::nth_element(buffer.begin(), buffer.begin() + N / 2, buffer.begin() + N);
    return buffer[N / 2];
}

/// \brief Calculate the Median Absolute Deviation statistic
///
/// \param buffer Container of scalar data to find MAD of.
/// \param median Precomputed median of buffer.
/// \return MAD = median(| X_i - median(X) |)
double getMAD(const std::vector<double>& buffer, double median) {
    const size_t N = buffer.size();
    std::vector<double> shifted(N);
    for (size_t i = 0; i < N; i++) {
        shifted[i] = std::abs(buffer[i] - median);
    }
    std::nth_element(shifted.begin(), shifted.begin() + N / 2, shifted.begin() + N);
    static constexpr double k = 1.4826; // assumes normally distributed data
    return k * shifted[N / 2];
}

/// \brief Calculate spread of data as interval around median
///
/// I = [min, max] = [median(X) - α·MAD, median(X) + α·MAD]
///
/// \param buffer Container of scalar data to find spread of.
/// \param min Alpha MADs below median.
/// \param max Alpha MADs above median.
void getMinMaxRScore(const std::vector<double>& buffer, double& min, double& max, double alpha) {
    double median = getMedian(buffer);
    double mad = getMAD(buffer, median);
    min = median - alpha * mad;
    max = median + alpha * mad;
}

void getConvexHull2D(const std::vector<Eigen::Vector2d>& points, std::vector<size_t>& indices) {
    static constexpr int DIM = 2;
    // auto convex_hull = std::make_shared<TriangleMesh>();
    std::vector<size_t> pt_map;

    std::vector<double> qhull_points_data(points.size() * DIM);
    for (size_t pidx = 0; pidx < points.size(); ++pidx) {
        const auto& pt = points[pidx];
        qhull_points_data[pidx * DIM + 0] = pt(0);
        qhull_points_data[pidx * DIM + 1] = pt(1);
    }

    orgQhull::PointCoordinates qhull_points(DIM, "");
    qhull_points.append(qhull_points_data);

    orgQhull::Qhull qhull;
    qhull.runQhull(qhull_points.comment().c_str(), qhull_points.dimension(),
                   qhull_points.count(), qhull_points.coordinates(), "");

    orgQhull::QhullVertexList vertices = qhull.vertexList();
    indices.clear();
    indices.reserve(vertices.size());
    for (orgQhull::QhullVertexList::iterator it = vertices.begin();
        it != vertices.end(); ++it) {
        indices.push_back(it->point().id());
    }
}

/// \class PlaneDetector
///
/// \brief Robust detection of planes from point sets
///
/// Using median as a robust estimator, PlaneDetector consumes a point cloud
/// and estimate a single plane using the point positions and their normals.
/// Planarity is verified by statistics-based tests on the associated points.
/// This implementation follows the work of [ArujoOliveira2020] as outlined in
///
///     Araújo and Oliveira, “A robust statistics approach for plane
///     detection in unorganized point clouds,” Pattern Recognition, 2020.
///
/// See also https://www.inf.ufrgs.br/~oliveira/pubs_files/RE/RE.html
class PlaneDetector
{
public:
    /// \brief Constructor to initialize detection parameters.
    ///
    /// \param normal_similarity is the min allowable similarity score
    /// between a point normal and the detected plane normal.
    /// \param coplanarity is the max allowable similiarity between
    /// detected plane normal and auxiliary planarity test vector. An
    /// ideal plane has score 0, i.e., normal orthogonal to test vector.
    /// \param outlier_ratio is the max allowable ratio of outlier points.
    PlaneDetector(double normal_similarity, double coplanarity, double outlier_ratio)
        : patch_(std::make_shared<PlanarPatch>()),
            normal_similarity_thr_(normal_similarity),
            coplanarity_thr_(coplanarity),
            outlier_ratio_thr_(outlier_ratio) {}
    ~PlaneDetector() = default;

    /// \brief Estimate plane from point cloud and selected point indices.
    ///
    /// \param point_cloud The point cloud (with points and normals).
    /// \param indices Point indices within point cloud to use.
    /// \return True if indices of point cloud pass robust planarity tests.
    bool DetectFromPointCloud(const PointCloud* point_cloud, const std::vector<size_t>& indices) {
        if (point_cloud->IsEmpty()) return false;

        // Hold a reference to the point cloud. This PlaneDetector
        // object shall be released before the PointCloud object.
        point_cloud_ = point_cloud;
        indices_ = indices;

        // TODO: check if there are enough points

        // estimate a plane from the relevant points
        EstimatePlane();
        // std::cout << "estimated..." << std::flush;

        // check that the estimated plane passes the robust planarity tests
        return RobustPlanarityTest();
    }

    /// \brief Delimit the plane using its perimeter points.
    ///
    /// \return A patch which is the bounded version of the plane.
    std::shared_ptr<PlanarPatch> DelimitPlane()
    {

        Eigen::Matrix3Xd M;
        GetPlanePerimeterPoints(M);

        Eigen::Matrix3d B;
        ConstructOrthogonalBasis(B);

        // Bisection search to find new rotated basis that
        // minimizes the area of bounded plane.
        double min_angled = 0;
        double max_angled = 90;
        static constexpr double ANGLE_RESOLUTION_DEG = 5;
        while (max_angled - min_angled > ANGLE_RESOLUTION_DEG) {

            const double mid = (max_angled + min_angled) / 2.;
            const double left = (min_angled + mid) / 2.;
            const double right = (max_angled + mid) / 2.;

            RotatedRect leftRect(M, B, left);
            RotatedRect rightRect(M, B, right);
            if (leftRect.area < rightRect.area) {
                max_angled = mid;
            } else {
                min_angled = mid;
            }
        }

        // Create the optimum basis found from bisection search
        const double theta = (min_angled + max_angled) / 2.;
        RotatedRect rect(M, B, theta);

        // Update the center of the patch
        patch_->center_ -= rect.B.col(0).dot(patch_->center_) * rect.B.col(0);
        patch_->center_ -= rect.B.col(1).dot(patch_->center_) * rect.B.col(1);
        patch_->center_ += (rect.bottom_left(0) + rect.top_right(0)) / 2. * rect.B.col(0);
        patch_->center_ += (rect.bottom_left(1) + rect.top_right(1)) / 2. * rect.B.col(1);

        // Scale basis to fit points
        const double sx = (rect.top_right.x() - rect.bottom_left.x()) / 2.;
        const double sy = (rect.top_right.y() - rect.bottom_left.y()) / 2.;
        patch_->basis_x_ = sx * rect.B.col(0);
        patch_->basis_y_ = sy * rect.B.col(1);

        return patch_;
    }

public:
    std::shared_ptr<PlanarPatch> patch_;
    const PointCloud* point_cloud_;
    std::vector<size_t> indices_;

private:
    /// Indicates if patch passed the planarity test
    bool valid_ = false;

    double min_normal_diff_;
    double max_point_dist_;

    Eigen::Vector3d min_bound_ =  Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d max_bound_ = -Eigen::Vector3d::Constant(std::numeric_limits<double>::max());

    /// Minimum allowable similarity score for point normal to plane normal.
    double normal_similarity_thr_;
    /// Maximum allowable similarity between normal and auxiliary planarity test vector.
    /// An ideal plane has score 0, i.e., normal orthogonal to test vector.
    double coplanarity_thr_;
    /// Maximum allowable outlier ratio
    double outlier_ratio_thr_;

    struct RotatedRect
    {
        Eigen::Matrix3d B;
        Eigen::Matrix3Xd M;
        double area;
        Eigen::Vector3d bottom_left;
        Eigen::Vector3d top_right;

        RotatedRect(const Eigen::Matrix3Xd& M, const Eigen::Matrix3d& B, double degrees) {
            Eigen::Matrix3d R;
            R = Eigen::AngleAxisd(degrees * M_PI / 180., Eigen::Vector3d::UnitZ());

            this->B = B * R;
            this->M = this->B.transpose() * M;
            bottom_left = this->M.rowwise().minCoeff();
            top_right = this->M.rowwise().maxCoeff();
            const double w = top_right(0) - bottom_left(0);
            const double h = top_right(1) - bottom_left(1);
            area = w * h;
        }
    };

    /// \brief Estimate plane from point cloud and selected point indices.
    void EstimatePlane() {
        // Calculate the median of the points and normals to estimate plane.
        const size_t N = indices_.size();
        std::vector<double> center_buf(N, 0);
        std::vector<double> normal_buf(N, 0);
        for (size_t d = 0; d < 3; d++) {
            // put data into buffer
            for (size_t i = 0; i < N; i++) {
                center_buf[i] = point_cloud_->points_[indices_[i]](d);
                normal_buf[i] = point_cloud_->normals_[indices_[i]](d);

                // Simultaneously calculate the bounds of the associated points
                min_bound_(d) = std::min(min_bound_(d), point_cloud_->points_[indices_[i]](d));
                max_bound_(d) = std::max(max_bound_(d), point_cloud_->points_[indices_[i]](d));
            }
            // compute the median along this dimension
            patch_->center_(d) = getMedian(center_buf);
            patch_->normal_(d) = getMedian(normal_buf);
        }
        patch_->normal_.normalize();

        // orthogonal distance to plane
        patch_->dist_from_origin_ = -patch_->normal_.dot(patch_->center_);
    }

    /// \brief Use robust statistics (i.e., median) to test planarity.
    ///
    /// Follows Sec 3.2 of [ArujoOliveira2020].
    ///
    /// \return True if passes tests.
    bool RobustPlanarityTest() {
        // Calculate statistics to robustly test planarity.
        const size_t N = indices_.size();
        // Stores point-to-plane distance of each associated point.
        std::vector<double> point_distances(N, 0);
        // Stores dot product (similarity) of each point normal to plane normal.
        std::vector<double> normal_similarities(N, 0);
        for (size_t i = 0; i < N; i++) {
            const Eigen::Vector3d& normal = point_cloud_->normals_[indices_[i]];
            const Eigen::Vector3d& position = point_cloud_->points_[indices_[i]];
            // similarity of estimated plane normal to point normal
            normal_similarities[i] = std::abs(patch_->normal_.dot(normal));
            // distance from estimated plane to point
            point_distances[i] = std::abs(patch_->normal_.dot(position) + patch_->dist_from_origin_);
        }

        // std::cout << "calc stats..." << std::flush;

        double tmp;
        // Use lower bound of the spread around the median as an indication
        // of how similar the point normals associated with the patch are.
        getMinMaxRScore(normal_similarities, min_normal_diff_, tmp, 3);
        // Use upper bound of the spread around the median as an indication
        // of how close the points associated with the patch are to the patch.
        getMinMaxRScore(point_distances, tmp, max_point_dist_, 3);

        // std::cout << "ok..." << std::flush;

        // Fail if too much "variance" in how similar point normals are to patch normal
        // std::cout << "normal (" << min_normal_diff_ << " > " << normal_similarity_thr_ << ") " << std::flush;
        // if (!IsNormalValid()) std::cout << "invalid normal (" << min_normal_diff_ << " > " << normal_similarity_thr_ << ") " << std::flush;
        if (!IsNormalValid()) return false;

        // Fail if too much "variance" in distances of points to patch
        // IsDistanceValid();
        // const double longest_edge = (max_bound_ - min_bound_).maxCoeff();
        // std::cout << "distance (" << last_dist_ << " < " << coplanarity_thr_ << ") [" << longest_edge << "]" << std::flush;
        // if (!IsDistanceValid()) std::cout << "@$@$@$ invalid @$@$@$" << std::flush;
        if (!IsDistanceValid()) return false;

        // Detect outliers, fail if too many
        std::unordered_map<size_t, bool> outliers;
        outliers.reserve(N);
        size_t num_outliers = 0;
        // std::cout << "outlier check..." << std::flush;
        for (size_t i = 0; i < N; i++) {
            const bool is_outlier = normal_similarities[i] < min_normal_diff_
                                    || point_distances[i] > max_point_dist_;
            outliers[indices_[i]] = is_outlier;
            num_outliers += static_cast<int>(is_outlier);
        }
        // if (num_outliers > N * outlier_ratio_thr_) std::cout << "too many outliers (" << num_outliers << " / " << N << ")..." << std::flush;
        if (num_outliers > N * outlier_ratio_thr_) return false;

        // Remove outliers
        // std::cout << "rm..." << indices_.size() << "..." << num_outliers << "..." << std::flush;
        // utility::LogInfo("indices {} outliers ", indices_,  outliers);
        if (num_outliers > 0) {
            indices_.erase(std::remove_if(indices_.begin(), indices_.end(), [&outliers](const size_t& idx) {
                return outliers[idx];
            }), indices_.end());
        }

        // std::cout << "ok..." << std::flush;
        return true;
    }

    /// \brief Check if detected plane normal is similar to point normals.
    inline bool IsNormalValid() const { return min_normal_diff_ > normal_similarity_thr_; }

    /// \brief Check if point distances from detected plane are reasonable.
    ///
    /// Constructs an auxiliary vector which captures
    /// coplanarity and curvature of points.
    bool IsDistanceValid() /*const*/ {
        // Orthogonal basis
        // Given a vector v1, calculate two orthogonal vectors v2, v3
        const Eigen::Vector3d U = Eigen::Vector3d(patch_->normal_.y() - patch_->normal_.z(), -patch_->normal_.x(), patch_->normal_.x()).normalized();
        // const Eigen::Vector3d V = patch_->normal_.cross(U).normalized();

        // Test point-to-plane distance w.r.t coplanarity of points.
        // See Fig. 4 of [ArujoOliveira2020].
        const double longest_edge = (max_bound_ - min_bound_).maxCoeff();
        const Eigen::Vector3d F = (U * longest_edge + patch_->normal_ * max_point_dist_).normalized();
        return std::abs(F.dot(patch_->normal_)) < coplanarity_thr_;
    }

    void GetPlanePerimeterPoints(Eigen::Matrix3Xd& M)
    {

        Eigen::Matrix3d B;
        ConstructOrthogonalBasis(B);

        // project each point onto the 2D span (x-y) of B
        std::vector<Eigen::Vector2d> projectedPoints2d(indices_.size());
        for (size_t i = 0; i < indices_.size(); i++) {
            const auto& p = point_cloud_->points_[indices_[i]];

            const double u = p.dot(B.col(0));
            const double v = p.dot(B.col(1));
            projectedPoints2d[i] << u, v;
        }

        std::vector<size_t> perimeter;
        getConvexHull2D(projectedPoints2d, perimeter);
        // utility::LogInfo("Plane with {} points has convex hull of size {}.", indices_.size(), perimeter.size());

        M = Eigen::Matrix3Xd(3, perimeter.size());
        for (size_t i = 0; i < perimeter.size(); i++) {
            M.col(i) = point_cloud_->points_[indices_[perimeter[i]]];
        }

    }

    /// \brief Builds an orthogonal basis of the plane using the normal for up.
    ///
    /// The constructed basis matrix can be interpreted as the rotation
    /// of the plane w.r.t the world (or, the point cloud sensor) frame.
    /// In other words, each column of B = [x^w_p y^w_p z^w_p] is one of
    /// the plane's basis vectors expressed in the world frame. Therefore,
    /// the matrix B ( = R^w_p) can be used to express the plane points M
    /// (expressed in the world frame) into the "plane" frame via M' = B^T*M
    ///
    /// \param B The 3x3 basis matrix.
    void ConstructOrthogonalBasis(Eigen::Matrix3d& B)
    {
        // construct x-vec by cross(normal, [0;1;1])
        // TODO: will break if plane normal is [0;1;1]. In that case
        // use a different vector to cross product with.
        B.col(0) = Eigen::Vector3d(patch_->normal_.y() - patch_->normal_.z(), -patch_->normal_.x(), patch_->normal_.x()).normalized();
        B.col(1) = patch_->normal_.cross(B.col(0)).normalized();
        B.col(2) = patch_->normal_;
    }
};

using BoundaryVolumeHierarchyPtr = std::shared_ptr<BoundaryVolumeHierarchy>;
using PlaneDetectorPtr = std::shared_ptr<PlaneDetector>;

bool SplitAndDetectPlanesRecursive(const BoundaryVolumeHierarchyPtr& node,
                                    size_t min_num_points,
                                    std::vector<PlaneDetectorPtr>& planes,
                                    double normal_similarity,
                                    double coplanarity,
                                    double outlier_ratio) {
    // if (node->indices().size() < min_num_points) {
    //     utility::LogInfo("Node {}.{} has too few points to partition or detect plane ({})", node->level_, node->child_index_, node->indices().size());
    // }


    // if there aren't enough points to find a good plane, don't even try
    if (node->indices().size() < min_num_points) return false;

    bool node_has_plane = false;
    bool child_has_plane = false;

    // partition into eight children and check each recursively for a plane
    // std::cout << std::string(node->level_, ' ') << "Partitioning node " << node->level_ << "." << node->child_index_ << "\t\t(" << node->indices().size() << ")" << std::flush;
    node->partition();
    // std::cout << "\t[" << std::flush;
    // for (size_t i=0; i<8; i++) if (node->children_[i]) std::cout << i << " " << std::flush;
    // std::cout << "]" << std::endl;

    for (const auto& child : node->children_) {
        if (child != nullptr &&
            SplitAndDetectPlanesRecursive(child, min_num_points, planes, normal_similarity, coplanarity, outlier_ratio)) {
            child_has_plane = true;
        }
    }

    if (!child_has_plane && node->level_ > 2) {
        auto plane = std::make_shared<PlaneDetector>(normal_similarity, coplanarity, outlier_ratio);
        if (plane->DetectFromPointCloud(node->point_cloud_, node->indices())) {
            node_has_plane = true;
            planes.push_back(plane);
            // std::cout << std::string(node->level_, ' ') << " ++ has a plane with " << node->indices().size() << " points" << std::flush;
            // std::cout << "\t--->\t " << plane->patch_->normal_.transpose() << " " << plane->patch_->dist_from_origin_ << std::endl;
        // } else {
            // std::cout << std::string(node->level_, ' ') << " -- has no plane" << std::endl;
        }
    }

    return node_has_plane || child_has_plane;
}

void ExtractPatchesFromPlanes(const std::vector<PlaneDetectorPtr>& planes, std::vector<std::shared_ptr<PlanarPatch>>& patches) {
    // Colors (default MATLAB colors)
    static constexpr int NUM_COLORS = 6;
    static std::array<Eigen::Vector3d, NUM_COLORS> colors_ = {
        Eigen::Vector3d(0.8500, 0.3250, 0.0980),
        Eigen::Vector3d(0.9290, 0.6940, 0.1250),
        Eigen::Vector3d(0.4940, 0.1840, 0.5560),
        Eigen::Vector3d(0.4660, 0.6740, 0.1880),
        Eigen::Vector3d(0.3010, 0.7450, 0.9330),
        Eigen::Vector3d(0.6350, 0.0780, 0.1840)
    };

    // std::cout << "++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    for (size_t i = 0; i < planes.size(); i++) {
        // create a patch by delimiting the plane using its perimeter points
        // std::cout << "Delimiting [" << plane->patch_->normal_.transpose() << " " << plane->patch_->dist_from_origin_ << "]\t" << std::flush;
        auto patch = planes[i]->DelimitPlane();
        // std::cout << std::endl;
        patch->PaintUniformColor(colors_[i%NUM_COLORS]);
        patches.push_back(patch);
    }
    // std::cout << "++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}

}  // unnamed namespace

std::vector<std::shared_ptr<PlanarPatch>> PointCloud::DetectPlanarPatches(double normal_similarity, double coplanarity, double outlier_ratio)
        const {
    if (!HasNormals()) {
        utility::LogError(
                "DetectPlanarPatches requires pre-computed normal vectors.");
        return {};
    }

    int min_num_points = 30;

    BoundaryVolumeHierarchyPtr root =
            std::make_shared<BoundaryVolumeHierarchy>(this);
    std::vector<PlaneDetectorPtr> planes;
    SplitAndDetectPlanesRecursive(root, min_num_points, planes, normal_similarity, coplanarity, outlier_ratio);

    // DetectPlanarPatchesRecursive

    // split phase

    // robust planarity test

    // grow phase

    // merge phase

    // iterative grow-merge

    std::vector<std::shared_ptr<PlanarPatch>> patches;
    ExtractPatchesFromPlanes(planes, patches);

    return patches;
}

}  // namespace geometry
}  // namespace open3d
