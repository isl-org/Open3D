// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>
#include <vector>

#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

const static std::vector<DataDescriptor> data_descriptors = {
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/livingroom.ply.zip",
         "841f32ff6294bb52d5f9574834e0925e"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/livingroom2-color.zip",
         "34792f7aa35b6c8b62e394e9372b95d7", "color"},
        {Open3DDownloadsPrefix() +
                 "augmented-icl-nuim/livingroom2-depth-clean.zip",
         "569c7ba065c3a84de32b0d0844699f43", "depth"},
        {Open3DDownloadsPrefix() +
                 "augmented-icl-nuim/livingroom2-depth-simulated.zip",
         "cbb4a36b8488448d79ec93c202c0c90e", "depth_noisy"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/livingroom2-traj.txt",
         "193961c018f4c2a753458d5231544036"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/livingroom2.oni.zip",
         "b827aa33cddffc8131d9b25007930137"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/dist-model.txt",
         "d8d7b6d29e754c2993a6eba4fd8d89ea"},
};

RedwoodIndoorLivingRoom2::RedwoodIndoorLivingRoom2(const std::string& data_root)
    : DownloadDataset("RedwoodIndoorLivingRoom2", data_descriptors, data_root) {
    const std::string extract_dir = GetExtractDir();
    std::vector<std::string> all_paths;

    // point_cloud_path_
    point_cloud_path_ = extract_dir + "/livingroom.ply";
    all_paths.push_back(point_cloud_path_);

    // color_paths_
    for (int i = 0; i <= 2349; i++) {
        const std::string path =
                extract_dir + "/color/" + fmt::format("{:05d}.jpg", i);
        color_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // depth_paths_
    for (int i = 0; i <= 2349; i++) {
        const std::string path =
                extract_dir + "/depth/" + fmt::format("{:05d}.png", i);
        depth_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // noisy_depth_paths_
    for (int i = 0; i <= 2349; i++) {
        const std::string path =
                extract_dir + "/depth_noisy/" + fmt::format("{:05d}.png", i);
        noisy_depth_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // oni_path_
    oni_path_ = extract_dir + "/livingroom2.oni";
    all_paths.push_back(oni_path_);

    // trajectory_path_
    trajectory_path_ = extract_dir + "/livingroom2-traj.txt";
    all_paths.push_back(trajectory_path_);

    // noise_model_path_
    noise_model_path_ = extract_dir + "/dist-model.txt";
    all_paths.push_back(noise_model_path_);

    // Check all files exist.
    CheckPathsExist(all_paths);
}

}  // namespace data
}  // namespace open3d
