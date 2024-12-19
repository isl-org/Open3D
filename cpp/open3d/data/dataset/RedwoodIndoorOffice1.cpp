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
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/office.ply.zip",
         "ba3640bba38f19c8f2d5e86e045eeae5"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/office1-color.zip",
         "6a58750880e83ac5948e0f28de294c04", "color"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/office1-depth-clean.zip",
         "0a952d68eb76e84fad63e362a59f82cd", "depth"},
        {Open3DDownloadsPrefix() +
                 "augmented-icl-nuim/office1-depth-simulated.zip",
         "7c7e479191d35c2ad1f8ac1c227f4f8d", "depth_noisy"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/office1-traj.txt",
         "3fac752ab38a4e8a96d1b5afa535f9f7"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/office1.oni.zip",
         "1edd52a60b052fde97b05ae3d628caba"},
        {Open3DDownloadsPrefix() + "augmented-icl-nuim/dist-model.txt",
         "d8d7b6d29e754c2993a6eba4fd8d89ea"},
};

RedwoodIndoorOffice1::RedwoodIndoorOffice1(const std::string& data_root)
    : DownloadDataset("RedwoodIndoorOffice1", data_descriptors, data_root) {
    const std::string extract_dir = GetExtractDir();
    std::vector<std::string> all_paths;

    // point_cloud_path_
    point_cloud_path_ = extract_dir + "/office.ply";
    all_paths.push_back(point_cloud_path_);

    // color_paths_
    for (int i = 0; i <= 2689; i++) {
        const std::string path =
                extract_dir + "/color/" + fmt::format("{:05d}.jpg", i);
        color_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // depth_paths_
    for (int i = 0; i <= 2689; i++) {
        const std::string path =
                extract_dir + "/depth/" + fmt::format("{:05d}.png", i);
        depth_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // noisy_depth_paths_
    for (int i = 0; i <= 2689; i++) {
        const std::string path =
                extract_dir + "/depth_noisy/" + fmt::format("{:05d}.png", i);
        noisy_depth_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // oni_path_
    oni_path_ = extract_dir + "/office1.oni";
    all_paths.push_back(oni_path_);

    // trajectory_path_
    trajectory_path_ = extract_dir + "/office1-traj.txt";
    all_paths.push_back(trajectory_path_);

    // noise_model_path_
    noise_model_path_ = extract_dir + "/dist-model.txt";
    all_paths.push_back(noise_model_path_);

    // Check all files exist.
    CheckPathsExist(all_paths);
}

}  // namespace data
}  // namespace open3d
