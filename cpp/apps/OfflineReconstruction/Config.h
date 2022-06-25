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

#include <json/json.h>

#include "FileSystemUtil.h"
#include "open3d/Open3D.h"

namespace open3d {
namespace apps {
namespace offline_reconstruction {

void SetDefaultValue(Json::Value& config,
                     const std::string& key,
                     int default_value) {
    if (!config.isMember(key)) {
        config[key] = default_value;
    }
}

void SetDefaultValue(Json::Value& config,
                     const std::string& key,
                     float default_value) {
    if (!config.isMember(key)) {
        config[key] = default_value;
    }
}

void SetDefaultValue(Json::Value& config,
                     const std::string& key,
                     const std::string& default_value) {
    if (!config.isMember(key)) {
        config[key] = default_value;
    }
}

void InitConfig(Json::Value& config) {
    // Set default parameters if not specified.
    SetDefaultValue(config, "n_frames_per_fragment", 100);
    SetDefaultValue(config, "n_keyframes_per_n_frame", 5);
    SetDefaultValue(config, "depth_min", 0.3f);
    SetDefaultValue(config, "depth_max", 3.0f);
    SetDefaultValue(config, "voxel_size", 0.05f);
    SetDefaultValue(config, "depth_diff_max", 0.07f);
    SetDefaultValue(config, "depth_scale", 1000);
    SetDefaultValue(config, "preference_loop_closure_odometry", 0.1f);
    SetDefaultValue(config, "preference_loop_closure_registration", 5.0f);
    SetDefaultValue(config, "tsdf_cubic_size", 3.0f);
    SetDefaultValue(config, "icp_method", "color");
    SetDefaultValue(config, "global_registration", "ransac");
    SetDefaultValue(config, "multi_threading", true);

    // `slac` and `slac_integrate` related parameters. `voxel_size` and
    // `depth_min` parameters from previous section, are also used in `slac`
    // and `slac_integrate`.
    SetDefaultValue(config, "max_iterations", 5);
    SetDefaultValue(config, "sdf_trunc", 0.04f);
    SetDefaultValue(config, "block_count", 40000);
    SetDefaultValue(config, "distance_threshold", 0.07f);
    SetDefaultValue(config, "fitness_threshold", 0.3f);
    SetDefaultValue(config, "regularizer_weight", 1);
    SetDefaultValue(config, "method", "slac");
    SetDefaultValue(config, "device", "CPU:0");
    SetDefaultValue(config, "save_output_as", "pointcloud");
    SetDefaultValue(config, "folder_slac", "slac/");
    SetDefaultValue(config, "template_optimized_posegraph_slac",
                    "optimized_posegraph_slac.json");

    // Path related parameters.
    SetDefaultValue(config, "folder_fragment", "fragments/");
    SetDefaultValue(
            config, "subfolder_slac",
            "slac/" + FloatToString(config["voxel_size"].asFloat(), 3) + "/");
    SetDefaultValue(config, "template_fragment_posegraph", "fragments/");
    SetDefaultValue(config, "template_fragment_posegraph_optimized",
                    "fragments/");
    SetDefaultValue(config, "template_fragment_pointcloud", "fragments/");
    SetDefaultValue(config, "folder_scene", "scene/");
    SetDefaultValue(config, "template_global_posegraph",
                    "scene/global_registration.json");
    SetDefaultValue(config, "template_global_posegraph_optimized",
                    "scene/global_registration_optimized.json");
    SetDefaultValue(config, "template_refined_posegraph",
                    "scene/refined_registration.json");
    SetDefaultValue(config, "template_refined_posegraph_optimized",
                    "scene/refined_registration_optimized.json");
    SetDefaultValue(config, "template_global_mesh", "scene/integrated.ply");
    SetDefaultValue(config, "template_global_traj", "scene/trajectory.log");

    if (utility::filesystem::GetFileExtensionInLowerCase(
                config["path_dataset"].asString()) == "bag") {
        std::tie(config["path_dataset"], config["path_intrinsic"],
                 config["depth_scale"]) =
                ExtractRGBDFrames(config["path_dataset"].asString());
    }
}

void LoungeDataLoader(Json::Value& config) {
    utility::LogInfo("Loading Stanford Lounge RGB-D Dataset");

    data::LoungeRGBDImages rgbd;

    // Set dataset specific parameters.
    config["path_dataset"] = rgbd.GetExtractDir();
    config["path_intrinsic"] = "";
    config["depth_max"] = 3.0f;
    config["voxel_size"] = 0.05f;
    config["depth_diff_max"] = 0.07f;
    config["preference_loop_closure_odometry"] = 0.1f;
    config["preference_loop_closure_registration"] = 5.0f;
    config["tsdf_cubic_size"] = 3.0f;
    config["icp_method"] = "color";
    config["global_registration"] = "ransac";
    config["multi_threading"] = true;
}

void BedroomDataLoader(Json::Value& config) {
    utility::LogInfo("Loading Redwood Bedroom RGB-D Dataset");

    data::BedroomRGBDImages rgbd;

    // Set dataset specific parameters.
    config["path_dataset"] = rgbd.GetExtractDir();
    config["path_intrinsic"] = "";
    config["depth_max"] = 3.0f;
    config["voxel_size"] = 0.05f;
    config["depth_diff_max"] = 0.07f;
    config["preference_loop_closure_odometry"] = 0.1f;
    config["preference_loop_closure_registration"] = 5.0f;
    config["tsdf_cubic_size"] = 3.0f;
    config["icp_method"] = "color";
    config["global_registration"] = "ransac";
    config["multi_threading"] = true;
}

void JackJackroomDataLoader(Json::Value& config) {
    utility::LogInfo("Loading RealSense L515 Jack-Jack RGB-D Dataset");

    data::JackJackL515Bag rgbd;

    // Set dataset specific parameters.
    config["path_dataset"] = rgbd.GetExtractDir();
    config["path_intrinsic"] = "";
    config["depth_max"] = 0.85f;
    config["voxel_size"] = 0.025f;
    config["depth_diff_max"] = 0.03f;
    config["preference_loop_closure_odometry"] = 0.1f;
    config["preference_loop_closure_registration"] = 5.0f;
    config["tsdf_cubic_size"] = 0.75f;
    config["icp_method"] = "color";
    config["global_registration"] = "ransac";
    config["multi_threading"] = true;
}

Json::Value DefaultDatasetLoader(const std::string& name) {
    utility::LogInfo("Config file was not passed. Using deafult dataset: {}.",
                     name);
    Json::Value config;
    if (name == "lounge") {
        LoungeDataLoader(config);
    } else if (name == "bedroom") {
        BedroomDataLoader(config);
    } else if (name == "jack_jack") {
        JackJackroomDataLoader(config);
    } else {
        utility::LogError("Dataset {} is not supported.", name);
    }

    InitConfig(config);
    utility::LogInfo("Loaded data from {}", config["path_dataset"].asString());

    return config;
}

}  // namespace offline_reconstruction
}  // namespace apps
}  // namespace open3d