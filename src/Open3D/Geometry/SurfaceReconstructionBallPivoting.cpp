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

#include "Open3D/Geometry/IntersectionTest.h"
#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

#include <Eigen/Dense>

#include <iostream>
#include <list>

namespace open3d {
namespace geometry {

class BallPivotingVertex;
class BallPivotingEdge;
class BallPivotingTriangle;

typedef BallPivotingVertex* BallPivotingVertexPtr;
typedef std::shared_ptr<BallPivotingEdge> BallPivotingEdgePtr;
typedef std::shared_ptr<BallPivotingTriangle> BallPivotingTrianglePtr;

class BallPivotingVertex {
public:
    enum Type { Orphan = 0, Front = 1, Inner = 2 };

    BallPivotingVertex(int idx,
                       const Eigen::Vector3d& point,
                       const Eigen::Vector3d& normal)
        : idx_(idx), point_(point), normal_(normal), type_(Orphan) {}

    void UpdateType();

public:
    int idx_;
    const Eigen::Vector3d& point_;
    const Eigen::Vector3d& normal_;
    std::unordered_set<BallPivotingEdgePtr> edges_;
    Type type_;
};

class BallPivotingEdge {
public:
    enum Type { Border = 0, Front = 1, Inner = 2 };

    BallPivotingEdge(BallPivotingVertexPtr source, BallPivotingVertexPtr target)
        : source_(source), target_(target), type_(Type::Front) {}

    void AddAdjacentTriangle(BallPivotingTrianglePtr triangle);
    BallPivotingVertexPtr GetOppositeVertex();

public:
    BallPivotingVertexPtr source_;
    BallPivotingVertexPtr target_;
    BallPivotingTrianglePtr triangle0_;
    BallPivotingTrianglePtr triangle1_;
    Type type_;
};

class BallPivotingTriangle {
public:
    BallPivotingTriangle(BallPivotingVertexPtr vert0,
                         BallPivotingVertexPtr vert1,
                         BallPivotingVertexPtr vert2,
                         Eigen::Vector3d ball_center)
        : vert0_(vert0),
          vert1_(vert1),
          vert2_(vert2),
          ball_center_(ball_center) {}

public:
    BallPivotingVertexPtr vert0_;
    BallPivotingVertexPtr vert1_;
    BallPivotingVertexPtr vert2_;
    Eigen::Vector3d ball_center_;
};

void BallPivotingVertex::UpdateType() {
    if (edges_.empty()) {
        type_ = Type::Orphan;
    } else {
        for (const BallPivotingEdgePtr& edge : edges_) {
            if (edge->type_ != BallPivotingEdge::Type::Inner) {
                type_ = Type::Front;
                return;
            }
        }
        type_ = Type::Inner;
    }
}

void BallPivotingEdge::AddAdjacentTriangle(BallPivotingTrianglePtr triangle) {
    if (triangle != triangle0_ && triangle != triangle1_) {
        if (triangle0_ == nullptr) {
            triangle0_ = triangle;
            type_ = Type::Front;
            // update orientation
            BallPivotingVertexPtr opp = GetOppositeVertex();
            Eigen::Vector3d tr_norm =
                    (target_->point_ - source_->point_)
                            .cross(opp->point_ - source_->point_);
            tr_norm /= tr_norm.norm();
            Eigen::Vector3d pt_norm =
                    source_->normal_ + target_->normal_ + opp->normal_;
            pt_norm /= pt_norm.norm();
            if (pt_norm.dot(tr_norm) < 0) {
                std::swap(target_, source_);
            }
        } else if (triangle1_ == nullptr) {
            triangle1_ = triangle;
            type_ = Type::Inner;
        } else {
            utility::LogDebug("!!! This case should not happen");
        }
    }
}

BallPivotingVertexPtr BallPivotingEdge::GetOppositeVertex() {
    if (triangle0_ != nullptr) {
        if (triangle0_->vert0_->idx_ != source_->idx_ &&
            triangle0_->vert0_->idx_ != target_->idx_) {
            return triangle0_->vert0_;
        } else if (triangle0_->vert1_->idx_ != source_->idx_ &&
                   triangle0_->vert1_->idx_ != target_->idx_) {
            return triangle0_->vert1_;
        } else {
            return triangle0_->vert2_;
        }
    } else {
        return nullptr;
    }
}

class BallPivoting {
public:
    BallPivoting(const PointCloud& pcd)
        : has_normals_(pcd.HasNormals()), kdtree_(pcd) {
        mesh_ = std::make_shared<TriangleMesh>();
        mesh_->vertices_ = pcd.points_;
        mesh_->vertex_normals_ = pcd.normals_;
        mesh_->vertex_colors_ = pcd.colors_;
        for (size_t vidx = 0; vidx < pcd.points_.size(); ++vidx) {
            vertices.emplace_back(new BallPivotingVertex(static_cast<int>(vidx),
                                                         pcd.points_[vidx],
                                                         pcd.normals_[vidx]));
        }
    }

    virtual ~BallPivoting() {
        for (auto vert : vertices) {
            delete vert;
        }
    }

    bool ComputeBallCenter(int vidx1,
                           int vidx2,
                           int vidx3,
                           double radius,
                           Eigen::Vector3d& center) {
        const Eigen::Vector3d& v1 = vertices[vidx1]->point_;
        const Eigen::Vector3d& v2 = vertices[vidx2]->point_;
        const Eigen::Vector3d& v3 = vertices[vidx3]->point_;
        double c = (v2 - v1).squaredNorm();
        double b = (v1 - v3).squaredNorm();
        double a = (v3 - v2).squaredNorm();

        double alpha = a * (b + c - a);
        double beta = b * (a + c - b);
        double gamma = c * (a + b - c);
        double abg = alpha + beta + gamma;

        if (abg < 1e-16) {
            return false;
        }

        alpha = alpha / abg;
        beta = beta / abg;
        gamma = gamma / abg;

        Eigen::Vector3d circ_center = alpha * v1 + beta * v2 + gamma * v3;
        double circ_radius2 = a * b * c;

        a = std::sqrt(a);
        b = std::sqrt(b);
        c = std::sqrt(c);
        circ_radius2 = circ_radius2 /
                       ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c));

        double height = radius * radius - circ_radius2;
        if (height >= 0.0) {
            Eigen::Vector3d tr_norm = (v2 - v1).cross(v3 - v1);
            tr_norm /= tr_norm.norm();
            Eigen::Vector3d pt_norm = vertices[vidx1]->normal_ +
                                      vertices[vidx2]->normal_ +
                                      vertices[vidx3]->normal_;
            pt_norm /= pt_norm.norm();
            if (tr_norm.dot(pt_norm) < 0) {
                tr_norm *= -1;
            }

            height = sqrt(height);
            center = circ_center + height * tr_norm;
            return true;
        }
        return false;
    }

    BallPivotingEdgePtr GetLinkingEdge(const BallPivotingVertexPtr& v0,
                                       const BallPivotingVertexPtr& v1) {
        for (BallPivotingEdgePtr edge0 : v0->edges_) {
            for (BallPivotingEdgePtr edge1 : v1->edges_) {
                if (edge0->source_->idx_ == edge1->source_->idx_ &&
                    edge0->target_->idx_ == edge1->target_->idx_) {
                    return edge0;
                }
            }
        }
        return nullptr;
    }

    void CreateTriangle(const BallPivotingVertexPtr& v0,
                        const BallPivotingVertexPtr& v1,
                        const BallPivotingVertexPtr& v2,
                        const Eigen::Vector3d& center) {
        utility::LogDebug(
                "[CreateTriangle] with v0.idx={}, v1.idx={}, v2.idx={}",
                v0->idx_, v1->idx_, v2->idx_);
        BallPivotingTrianglePtr triangle =
                std::make_shared<BallPivotingTriangle>(v0, v1, v2, center);

        BallPivotingEdgePtr e0 = GetLinkingEdge(v0, v1);
        if (e0 == nullptr) {
            e0 = std::make_shared<BallPivotingEdge>(v0, v1);
        }
        e0->AddAdjacentTriangle(triangle);
        v0->edges_.insert(e0);
        v1->edges_.insert(e0);

        BallPivotingEdgePtr e1 = GetLinkingEdge(v1, v2);
        if (e1 == nullptr) {
            e1 = std::make_shared<BallPivotingEdge>(v1, v2);
        }
        e1->AddAdjacentTriangle(triangle);
        v1->edges_.insert(e1);
        v2->edges_.insert(e1);

        BallPivotingEdgePtr e2 = GetLinkingEdge(v2, v0);
        if (e2 == nullptr) {
            e2 = std::make_shared<BallPivotingEdge>(v2, v0);
        }
        e2->AddAdjacentTriangle(triangle);
        v2->edges_.insert(e2);
        v0->edges_.insert(e2);

        v0->UpdateType();
        v1->UpdateType();
        v2->UpdateType();

        Eigen::Vector3d face_normal =
                ComputeFaceNormal(v0->point_, v1->point_, v2->point_);
        if (face_normal.dot(v0->normal_) > -1e-16) {
            mesh_->triangles_.emplace_back(
                    Eigen::Vector3i(v0->idx_, v1->idx_, v2->idx_));
        } else {
            mesh_->triangles_.emplace_back(
                    Eigen::Vector3i(v0->idx_, v2->idx_, v1->idx_));
        }
        mesh_->triangle_normals_.push_back(face_normal);
    }

    Eigen::Vector3d ComputeFaceNormal(const Eigen::Vector3d& v0,
                                      const Eigen::Vector3d& v1,
                                      const Eigen::Vector3d& v2) {
        Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0);
        double norm = normal.norm();
        if (norm > 0) {
            normal /= norm;
        }
        return normal;
    }

    bool IsCompatible(const BallPivotingVertexPtr& v0,
                      const BallPivotingVertexPtr& v1,
                      const BallPivotingVertexPtr& v2) {
        utility::LogDebug("[IsCompatible] v0.idx={}, v1.idx={}, v2.idx={}",
                          v0->idx_, v1->idx_, v2->idx_);
        Eigen::Vector3d normal =
                ComputeFaceNormal(v0->point_, v1->point_, v2->point_);
        if (normal.dot(v0->normal_) < -1e-16) {
            normal *= -1;
        }
        bool ret = normal.dot(v0->normal_) > -1e-16 &&
                   normal.dot(v1->normal_) > -1e-16 &&
                   normal.dot(v2->normal_) > -1e-16;
        utility::LogDebug("[IsCompatible] retuns = {}", ret);
        return ret;
    }

    BallPivotingVertexPtr FindCandidateVertex(
            const BallPivotingEdgePtr& edge,
            double radius,
            Eigen::Vector3d& candidate_center) {
        utility::LogDebug("[FindCandidateVertex] edge=({}, {}), radius={}",
                          edge->source_->idx_, edge->target_->idx_, radius);
        BallPivotingVertexPtr src = edge->source_;
        BallPivotingVertexPtr tgt = edge->target_;

        const BallPivotingVertexPtr opp = edge->GetOppositeVertex();
        utility::LogDebug("[FindCandidateVertex] edge=({}, {}), opp={}",
                          src->idx_, tgt->idx_, opp->idx_);
        utility::LogDebug("[FindCandidateVertex] src={} => {}", src->idx_,
                          src->point_.transpose());
        utility::LogDebug("[FindCandidateVertex] tgt={} => {}", tgt->idx_,
                          tgt->point_.transpose());
        utility::LogDebug("[FindCandidateVertex] src={} => {}", opp->idx_,
                          opp->point_.transpose());

        Eigen::Vector3d mp = 0.5 * (src->point_ + tgt->point_);
        utility::LogDebug("[FindCandidateVertex] edge=({}, {}), mp={}",
                          edge->source_->idx_, edge->target_->idx_,
                          mp.transpose());

        BallPivotingTrianglePtr triangle = edge->triangle0_;
        const Eigen::Vector3d& center = triangle->ball_center_;
        utility::LogDebug("[FindCandidateVertex] edge=({}, {}), center={}",
                          edge->source_->idx_, edge->target_->idx_,
                          center.transpose());

        Eigen::Vector3d v = tgt->point_ - src->point_;
        v /= v.norm();

        Eigen::Vector3d a = center - mp;
        a /= a.norm();

        std::vector<int> indices;
        std::vector<double> dists2;
        kdtree_.SearchRadius(mp, 2 * radius, indices, dists2);
        utility::LogDebug("[FindCandidateVertex] found {} potential candidates",
                          indices.size());

        BallPivotingVertexPtr min_candidate = nullptr;
        double min_angle = 2 * M_PI;
        for (auto nbidx : indices) {
            utility::LogDebug("[FindCandidateVertex] nbidx {:d}", nbidx);
            const BallPivotingVertexPtr& candidate = vertices[nbidx];
            if (candidate->idx_ == src->idx_ || candidate->idx_ == tgt->idx_ ||
                candidate->idx_ == opp->idx_) {
                utility::LogDebug(
                        "[FindCandidateVertex] candidate {:d} is a triangle "
                        "vertex of the edge",
                        candidate->idx_);
                continue;
            }
            utility::LogDebug("[FindCandidateVertex] candidate={:d} => {}",
                              candidate->idx_, candidate->point_.transpose());

            bool coplanar = IntersectionTest::PointsCoplanar(
                    src->point_, tgt->point_, opp->point_, candidate->point_);
            if (coplanar && (IntersectionTest::LineSegmentsMinimumDistance(
                                     mp, candidate->point_, src->point_,
                                     opp->point_) < 1e-12 ||
                             IntersectionTest::LineSegmentsMinimumDistance(
                                     mp, candidate->point_, tgt->point_,
                                     opp->point_) < 1e-12)) {
                utility::LogDebug(
                        "[FindCandidateVertex] candidate {:d} is interesecting "
                        "the existing triangle",
                        candidate->idx_);
                continue;
            }

            Eigen::Vector3d new_center;
            if (!ComputeBallCenter(src->idx_, tgt->idx_, candidate->idx_,
                                   radius, new_center)) {
                utility::LogDebug(
                        "[FindCandidateVertex] candidate {:d} can not compute "
                        "ball",
                        candidate->idx_);
                continue;
            }
            utility::LogDebug("[FindCandidateVertex] candidate {:d} center={}",
                              candidate->idx_, new_center.transpose());

            Eigen::Vector3d b = new_center - mp;
            b /= b.norm();
            utility::LogDebug(
                    "[FindCandidateVertex] candidate {:d} v={}, a={}, b={}",
                    candidate->idx_, v.transpose(), a.transpose(),
                    b.transpose());

            double cosinus = a.dot(b);
            cosinus = std::min(cosinus, 1.0);
            cosinus = std::max(cosinus, -1.0);
            utility::LogDebug(
                    "[FindCandidateVertex] candidate {:d} cosinus={:f}",
                    candidate->idx_, cosinus);

            double angle = std::acos(cosinus);

            Eigen::Vector3d c = a.cross(b);
            if (c.dot(v) < 0) {
                angle = 2 * M_PI - angle;
            }

            if (angle >= min_angle) {
                utility::LogDebug(
                        "[FindCandidateVertex] candidate {:d} angle {:f} > "
                        "min_angle {:f}",
                        candidate->idx_, angle, min_angle);
                continue;
            }

            bool empty_ball = true;
            for (auto nbidx2 : indices) {
                const BallPivotingVertexPtr& nb = vertices[nbidx2];
                if (nb->idx_ == src->idx_ || nb->idx_ == tgt->idx_ ||
                    nb->idx_ == candidate->idx_) {
                    continue;
                }
                if ((new_center - nb->point_).norm() < radius - 1e-16) {
                    utility::LogDebug(
                            "[FindCandidateVertex] candidate {:d} not an empty "
                            "ball",
                            candidate->idx_);
                    empty_ball = false;
                    break;
                }
            }

            if (empty_ball) {
                utility::LogDebug("[FindCandidateVertex] candidate {:d} works",
                                  candidate->idx_);
                min_angle = angle;
                min_candidate = vertices[nbidx];
                candidate_center = new_center;
            }
        }

        if (min_candidate == nullptr) {
            utility::LogDebug("[FindCandidateVertex] returns nullptr");
        } else {
            utility::LogDebug("[FindCandidateVertex] returns {:d}",
                              min_candidate->idx_);
        }
        return min_candidate;
    }

    void ExpandTriangulation(double radius) {
        utility::LogDebug("[ExpandTriangulation] radius={}", radius);
        while (!edge_front_.empty()) {
            BallPivotingEdgePtr edge = edge_front_.front();
            edge_front_.pop_front();
            if (edge->type_ != BallPivotingEdge::Front) {
                continue;
            }

            Eigen::Vector3d center;
            BallPivotingVertexPtr candidate =
                    FindCandidateVertex(edge, radius, center);
            if (candidate == nullptr ||
                candidate->type_ == BallPivotingVertex::Type::Inner ||
                !IsCompatible(candidate, edge->source_, edge->target_)) {
                edge->type_ = BallPivotingEdge::Type::Border;
                border_edges_.push_back(edge);
                continue;
            }

            BallPivotingEdgePtr e0 = GetLinkingEdge(candidate, edge->source_);
            BallPivotingEdgePtr e1 = GetLinkingEdge(candidate, edge->target_);
            if ((e0 != nullptr && e0->type_ != BallPivotingEdge::Type::Front) ||
                (e1 != nullptr && e1->type_ != BallPivotingEdge::Type::Front)) {
                edge->type_ = BallPivotingEdge::Type::Border;
                border_edges_.push_back(edge);
                continue;
            }

            CreateTriangle(edge->source_, edge->target_, candidate, center);

            e0 = GetLinkingEdge(candidate, edge->source_);
            e1 = GetLinkingEdge(candidate, edge->target_);
            if (e0->type_ == BallPivotingEdge::Type::Front) {
                edge_front_.push_front(e0);
            }
            if (e1->type_ == BallPivotingEdge::Type::Front) {
                edge_front_.push_front(e1);
            }
        }
    }

    bool TryTriangleSeed(const BallPivotingVertexPtr& v0,
                         const BallPivotingVertexPtr& v1,
                         const BallPivotingVertexPtr& v2,
                         const std::vector<int>& nb_indices,
                         double radius,
                         Eigen::Vector3d& center) {
        utility::LogDebug(
                "[TryTriangleSeed] v0.idx={}, v1.idx={}, v2.idx={}, "
                "radius={}",
                v0->idx_, v1->idx_, v2->idx_, radius);

        if (!IsCompatible(v0, v1, v2)) {
            return false;
        }

        BallPivotingEdgePtr e0 = GetLinkingEdge(v0, v2);
        BallPivotingEdgePtr e1 = GetLinkingEdge(v1, v2);
        if (e0 != nullptr && e0->type_ == BallPivotingEdge::Type::Inner) {
            utility::LogDebug(
                    "[TryTriangleSeed] returns {} because e0 is inner edge",
                    false);
            return false;
        }
        if (e1 != nullptr && e1->type_ == BallPivotingEdge::Type::Inner) {
            utility::LogDebug(
                    "[TryTriangleSeed] returns {} because e1 is inner edge",
                    false);
            return false;
        }

        if (!ComputeBallCenter(v0->idx_, v1->idx_, v2->idx_, radius, center)) {
            utility::LogDebug(
                    "[TryTriangleSeed] returns {} could not compute ball "
                    "center",
                    false);
            return false;
        }

        // test if no other point is within the ball
        for (const auto& nbidx : nb_indices) {
            const BallPivotingVertexPtr& v = vertices[nbidx];
            if (v->idx_ == v0->idx_ || v->idx_ == v1->idx_ ||
                v->idx_ == v2->idx_) {
                continue;
            }
            if ((center - v->point_).norm() < radius - 1e-16) {
                utility::LogDebug(
                        "[TryTriangleSeed] returns {} computed ball is not "
                        "empty",
                        false);
                return false;
            }
        }

        utility::LogDebug("[TryTriangleSeed] returns {}", true);
        return true;
    }

    bool TrySeed(BallPivotingVertexPtr& v, double radius) {
        utility::LogDebug("[TrySeed] with v.idx={}, radius={}", v->idx_,
                          radius);
        std::vector<int> indices;
        std::vector<double> dists2;
        kdtree_.SearchRadius(v->point_, 2 * radius, indices, dists2);
        if (indices.size() < 3u) {
            return false;
        }

        for (size_t nbidx0 = 0; nbidx0 < indices.size(); ++nbidx0) {
            const BallPivotingVertexPtr& nb0 = vertices[indices[nbidx0]];
            if (nb0->type_ != BallPivotingVertex::Type::Orphan) {
                continue;
            }
            if (nb0->idx_ == v->idx_) {
                continue;
            }

            int candidate_vidx2 = -1;
            Eigen::Vector3d center;
            for (size_t nbidx1 = nbidx0 + 1; nbidx1 < indices.size();
                 ++nbidx1) {
                const BallPivotingVertexPtr& nb1 = vertices[indices[nbidx1]];
                if (nb1->type_ != BallPivotingVertex::Type::Orphan) {
                    continue;
                }
                if (nb1->idx_ == v->idx_) {
                    continue;
                }
                if (TryTriangleSeed(v, nb0, nb1, indices, radius, center)) {
                    candidate_vidx2 = nb1->idx_;
                    break;
                }
            }

            if (candidate_vidx2 >= 0) {
                const BallPivotingVertexPtr& nb1 = vertices[candidate_vidx2];

                BallPivotingEdgePtr e0 = GetLinkingEdge(v, nb1);
                if (e0 != nullptr &&
                    e0->type_ != BallPivotingEdge::Type::Front) {
                    continue;
                }
                BallPivotingEdgePtr e1 = GetLinkingEdge(nb0, nb1);
                if (e1 != nullptr &&
                    e1->type_ != BallPivotingEdge::Type::Front) {
                    continue;
                }
                BallPivotingEdgePtr e2 = GetLinkingEdge(v, nb0);
                if (e2 != nullptr &&
                    e2->type_ != BallPivotingEdge::Type::Front) {
                    continue;
                }

                CreateTriangle(v, nb0, nb1, center);

                e0 = GetLinkingEdge(v, nb1);
                e1 = GetLinkingEdge(nb0, nb1);
                e2 = GetLinkingEdge(v, nb0);
                if (e0->type_ == BallPivotingEdge::Type::Front) {
                    edge_front_.push_front(e0);
                }
                if (e1->type_ == BallPivotingEdge::Type::Front) {
                    edge_front_.push_front(e1);
                }
                if (e2->type_ == BallPivotingEdge::Type::Front) {
                    edge_front_.push_front(e2);
                }

                if (edge_front_.size() > 0) {
                    utility::LogDebug(
                            "[TrySeed] edge_front_.size() > 0 => return "
                            "true");
                    return true;
                }
            }
        }

        utility::LogDebug("[TrySeed] return false");
        return false;
    }

    void FindSeedTriangle(double radius) {
        for (size_t vidx = 0; vidx < vertices.size(); ++vidx) {
            utility::LogDebug("[FindSeedTriangle] with radius={}, vidx={}",
                              radius, vidx);
            if (vertices[vidx]->type_ == BallPivotingVertex::Type::Orphan) {
                if (TrySeed(vertices[vidx], radius)) {
                    ExpandTriangulation(radius);
                }
            }
        }
    }

    std::shared_ptr<TriangleMesh> Run(const std::vector<double>& radii) {
        if (!has_normals_) {
            utility::LogError("ReconstructBallPivoting requires normals");
        }

        mesh_->triangles_.clear();

        for (double radius : radii) {
            utility::LogDebug("[Run] ################################");
            utility::LogDebug("[Run] change to radius {:.4f}", radius);
            if (radius <= 0) {
                utility::LogError(
                        "got an invalid, negative radius as parameter");
            }

            // update radius => update border edges
            for (auto it = border_edges_.begin(); it != border_edges_.end();) {
                BallPivotingEdgePtr edge = *it;
                BallPivotingTrianglePtr triangle = edge->triangle0_;
                utility::LogDebug(
                        "[Run] try edge {:d}-{:d} of triangle {:d}-{:d}-{:d}",
                        edge->source_->idx_, edge->target_->idx_,
                        triangle->vert0_->idx_, triangle->vert1_->idx_,
                        triangle->vert2_->idx_);

                Eigen::Vector3d center;
                if (ComputeBallCenter(triangle->vert0_->idx_,
                                      triangle->vert1_->idx_,
                                      triangle->vert2_->idx_, radius, center)) {
                    utility::LogDebug("[Run]   yes, we can work on this");
                    std::vector<int> indices;
                    std::vector<double> dists2;
                    kdtree_.SearchRadius(center, radius, indices, dists2);
                    bool empty_ball = true;
                    for (auto idx : indices) {
                        if (idx != triangle->vert0_->idx_ &&
                            idx != triangle->vert1_->idx_ &&
                            idx != triangle->vert2_->idx_) {
                            utility::LogDebug(
                                    "[Run]   but no, the ball is not empty");
                            empty_ball = false;
                            break;
                        }
                    }

                    if (empty_ball) {
                        utility::LogDebug(
                                "[Run]   yeah, add edge to edge_front_: {:d}",
                                edge_front_.size());
                        edge->type_ = BallPivotingEdge::Type::Front;
                        edge_front_.push_back(edge);
                        it = border_edges_.erase(it);
                        continue;
                    }
                }
                ++it;
            }

            // do the reconstruction
            if (edge_front_.empty()) {
                FindSeedTriangle(radius);
            } else {
                ExpandTriangulation(radius);
            }

            utility::LogDebug("[Run] mesh_ has {:d} triangles",
                              mesh_->triangles_.size());
            utility::LogDebug("[Run] ################################");
        }
        return mesh_;
    }

private:
    bool has_normals_;
    KDTreeFlann kdtree_;
    std::list<BallPivotingEdgePtr> edge_front_;
    std::list<BallPivotingEdgePtr> border_edges_;
    std::vector<BallPivotingVertexPtr> vertices;
    std::shared_ptr<TriangleMesh> mesh_;
};

std::shared_ptr<TriangleMesh> TriangleMesh::CreateFromPointCloudBallPivoting(
        const PointCloud& pcd, const std::vector<double>& radii) {
    BallPivoting bp(pcd);
    return bp.Run(radii);
}

}  // namespace geometry
}  // namespace open3d
