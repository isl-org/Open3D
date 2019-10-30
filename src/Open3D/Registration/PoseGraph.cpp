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

#include "Open3D/Registration/PoseGraph.h"

#include <json/json.h>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace registration {

PoseGraphNode::~PoseGraphNode() {}

bool PoseGraphNode::ConvertToJsonValue(Json::Value &value) const {
    value["class_name"] = "PoseGraphNode";
    value["version_major"] = 1;
    value["version_minor"] = 0;

    Json::Value pose_object;
    if (EigenMatrix4dToJsonArray(pose_, pose_object) == false) {
        return false;
    }
    value["pose"] = pose_object;
    return true;
}

bool PoseGraphNode::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "PoseGraphNode read JSON failed: unsupported json format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PoseGraphNode" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PoseGraphNode read JSON failed: unsupported json format.");
        return false;
    }

    const Json::Value &pose_object = value["pose"];
    if (EigenMatrix4dFromJsonArray(pose_, pose_object) == false) {
        return false;
    }
    return true;
}

PoseGraphEdge::~PoseGraphEdge() {}

bool PoseGraphEdge::ConvertToJsonValue(Json::Value &value) const {
    value["class_name"] = "PoseGraphEdge";
    value["version_major"] = 1;
    value["version_minor"] = 0;

    value["source_node_id"] = source_node_id_;
    value["target_node_id"] = target_node_id_;
    value["uncertain"] = uncertain_;
    value["confidence"] = confidence_;
    Json::Value transformation_object;
    if (EigenMatrix4dToJsonArray(transformation_, transformation_object) ==
        false) {
        return false;
    }
    value["transformation"] = transformation_object;
    Json::Value information_object;
    if (EigenMatrix6dToJsonArray(information_, information_object) == false) {
        return false;
    }
    value["information"] = information_object;
    return true;
}

bool PoseGraphEdge::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "PoseGraphEdge read JSON failed: unsupported json format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PoseGraphEdge" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PoseGraphEdge read JSON failed: unsupported json format.");
        return false;
    }

    source_node_id_ = value.get("source_node_id", -1).asInt();
    target_node_id_ = value.get("target_node_id", -1).asInt();
    uncertain_ = value.get("uncertain", false).asBool();
    confidence_ = value.get("confidence", 1.0).asDouble();
    const Json::Value &transformation_object = value["transformation"];
    if (EigenMatrix4dFromJsonArray(transformation_, transformation_object) ==
        false) {
        return false;
    }
    const Json::Value &information_object = value["information"];
    if (EigenMatrix6dFromJsonArray(information_, information_object) == false) {
        return false;
    }
    return true;
}

PoseGraph::PoseGraph() {}

PoseGraph::~PoseGraph() {}

bool PoseGraph::ConvertToJsonValue(Json::Value &value) const {
    value["class_name"] = "PoseGraph";
    value["version_major"] = 1;
    value["version_minor"] = 0;

    Json::Value node_array;
    for (const auto &node : nodes_) {
        Json::Value node_object;
        if (node.ConvertToJsonValue(node_object) == false) {
            return false;
        }
        node_array.append(node_object);
    }
    value["nodes"] = node_array;

    Json::Value edge_array;
    for (const auto &edge : edges_) {
        Json::Value edge_object;
        if (edge.ConvertToJsonValue(edge_object) == false) {
            return false;
        }
        edge_array.append(edge_object);
    }
    value["edges"] = edge_array;
    return true;
}

bool PoseGraph::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "PoseGraph read JSON failed: unsupported json format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PoseGraph" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PoseGraph read JSON failed: unsupported json format.");
        return false;
    }

    const Json::Value &node_array = value["nodes"];
    if (node_array.size() == 0) {
        utility::LogWarning("PoseGraph read JSON failed: empty nodes.");
        return false;
    }
    nodes_.clear();
    for (int i = 0; i < (int)node_array.size(); i++) {
        const Json::Value &node_object = node_array[i];
        PoseGraphNode new_node;
        if (new_node.ConvertFromJsonValue(node_object) == false) {
            return false;
        }
        nodes_.push_back(new_node);
    }

    const Json::Value &edge_array = value["edges"];
    if (edge_array.size() == 0) {
        utility::LogWarning("PoseGraph read JSON failed: empty edges.");
        return false;
    }
    edges_.clear();
    for (int i = 0; i < (int)edge_array.size(); i++) {
        const Json::Value &edge_object = edge_array[i];
        PoseGraphEdge new_edge;
        if (new_edge.ConvertFromJsonValue(edge_object) == false) {
            return false;
        }
        edges_.push_back(new_edge);
    }
    return true;
}

}  // namespace registration
}  // namespace open3d
