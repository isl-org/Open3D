// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
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

#include "PoseGraph.h"

#include <json/json.h>
#include <Core/Utility/Console.h>

namespace three{

PoseGraph::PoseGraph()
{
}

PoseGraph::~PoseGraph()
{
}

bool PoseGraph::ConvertToJsonValue(Json::Value &value) const
{
	value["class_name"] = "PoseGraph";
	value["version_major"] = 1;
	value["version_minor"] = 0;

	Json::Value node_array;
	for (const auto &status : nodes_) {
		Json::Value status_object;
		if (EigenMatrix4dToJsonArray(status.pose_, status_object) == false) {
			return false;
		}
		node_array.append(status_object);
	}	
	value["nodes"] = node_array;	

	Json::Value edge_array;
	for (const auto &status : edges_) {
		Json::Value status_object;
		value.append((double)status.target_node_id_);
		value.append((double)status.source_node_id_);
		if (EigenMatrix4dToJsonArray(
				status.transformation_, status_object) == false) {
			return false;
		}
		edge_array.append(status_object);
		if (EigenMatrix6dToJsonArray(
				status.information_, status_object) == false) {
			return false;
		}
		edge_array.append(status_object);
	}
	value["edges"] = edge_array;
	return true;
}

bool PoseGraph::ConvertFromJsonValue(const Json::Value &value)
{
	if (value.isObject() == false) {
		PrintWarning("PoseGraph read JSON failed: unsupported json format.\n");
		return false;		
	}
	if (value.get("class_name", "").asString() != "PoseGraph" ||
			value.get("version_major", 1).asInt() != 1 ||
			value.get("version_minor", 0).asInt() != 0) {
		PrintWarning("PoseGraph read JSON failed: unsupported json format.\n");
		return false;
	}

	const Json::Value &node_array = value["nodes"];
	if (node_array.size() == 0) {
		PrintWarning("PoseGraph read JSON failed: empty nodes.\n");
		return false;
	}
	nodes_.resize(node_array.size());
	for (int i = 0; i < (int)node_array.size(); i++) {
		const Json::Value &status_object = node_array[i];
		if (EigenMatrix4dFromJsonArray(nodes_[i].pose_, status_object) == false) {
			return false;
		}
	}

	const Json::Value &edge_array = value["edges"];
	if (edge_array.size() == 0) {
		PrintWarning("PoseGraph read JSON failed: empty edges.\n");
		return false;
	}
	edges_.resize(edge_array.size());
	for (int i = 0; i < (int)edge_array.size(); i++) {
		const Json::Value &status_object = edge_array[i];
		edges_[i].target_node_id_ = (int)status_object[0].asDouble();
		edges_[i].source_node_id_ = (int)status_object[1].asDouble();
		if (EigenMatrix4dFromJsonArray(edges_[i].transformation_, status_object[2]) == false) {
			return false;
		}
		if (EigenMatrix6dFromJsonArray(edges_[i].information_, status_object[18]) == false) {
			return false;
		}
	}
	
	return true;
}

}	// namespace three
