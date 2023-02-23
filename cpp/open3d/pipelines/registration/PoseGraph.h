// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "open3d/utility/Eigen.h"
#include "open3d/utility/IJsonConvertible.h"

namespace open3d {
namespace pipelines {
namespace registration {

/// \class PoseGraphNode
///
/// \brief Node of PoseGraph.
class PoseGraphNode : public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    PoseGraphNode(const Eigen::Matrix4d &pose = Eigen::Matrix4d::Identity())
        : pose_(pose) {}
    ~PoseGraphNode();

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    Eigen::Matrix4d_u pose_;
};

/// \class PoseGraphEdge
///
/// \brief Edge of PoseGraph.
class PoseGraphEdge : public utility::IJsonConvertible {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param source_node_id Source PoseGraphNode id.
    /// \param target_node_id Target PoseGraphNode id.
    /// \param transformation Transformation matrix.
    /// \param information Information matrix.
    /// \param uncertain Whether the edge is uncertain.
    /// \param confidence Confidence value of the edge.
    PoseGraphEdge(
            int source_node_id = -1,
            int target_node_id = -1,
            const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity(),
            const Eigen::Matrix6d &information = Eigen::Matrix6d::Identity(),
            bool uncertain = false,
            double confidence = 1.0)
        : source_node_id_(source_node_id),
          target_node_id_(target_node_id),
          transformation_(transformation),
          information_(information),
          uncertain_(uncertain),
          confidence_(confidence) {}
    ~PoseGraphEdge();

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// Source PoseGraphNode id.
    int source_node_id_;
    /// Target PoseGraphNode id.
    int target_node_id_;
    /// Transformation matrix.
    Eigen::Matrix4d_u transformation_;
    /// Information matrix.
    Eigen::Matrix6d_u information_;
    /// \brief Whether the edge is uncertain.
    ///
    /// Odometry edge has uncertain == false
    /// loop closure edges has uncertain == true
    bool uncertain_;
    /// \brief Confidence value of the edge.
    ///
    /// if uncertain_ is true, it has confidence bounded in [0,1].
    /// 1 means reliable, and 0 means unreliable edge.
    /// This correspondence to line process value in [Choi et al 2015]
    /// See core/registration/globaloptimization.h for more details.
    double confidence_;
};

/// \class PoseGraph
///
/// \brief Data structure defining the pose graph.
class PoseGraph : public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    PoseGraph();
    ~PoseGraph() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// List of PoseGraphNode.
    std::vector<PoseGraphNode> nodes_;
    /// List of PoseGraphEdge.
    std::vector<PoseGraphEdge> edges_;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
