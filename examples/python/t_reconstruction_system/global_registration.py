# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import tqdm as tqdm
import open3d as o3d
from config import ConfigParser

from common import get_default_dataset, load_fragment_file_names

if __name__ == "__main__":
    parser = ConfigParser()
    parser.add(
        "--config",
        is_config_file=True,
        help="YAML config file path."
        "Please refer to config.py for the options,"
        "and default_config.yml for default settings "
        "It overrides the default config file, but will be "
        "overridden by other command line inputs.",
    )
    parser.add(
        "--default_dataset",
        help="Default dataset is used when config file is not provided. "
        "Default dataset may be selected from the following options: "
        "[lounge, jack_jack]",
        default="lounge",
    )
    config = parser.get_config()

    if config.path_dataset == "":
        config = get_default_dataset(config)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    fragment_file_names = load_fragment_file_names(config)

    pcd0 = o3d.t.io.read_point_cloud(fragment_file_names[0]).cuda()
    pcd1 = o3d.t.io.read_point_cloud(fragment_file_names[5]).cuda()

    fpfh0 = o3d.t.pipelines.registration.compute_fpfh_feature(pcd0)
    fpfh1 = o3d.t.pipelines.registration.compute_fpfh_feature(pcd1)

    result = o3d.t.pipelines.registration.ransac_from_features(
        pcd0,
        pcd1,
        fpfh0,
        fpfh1,
        max_correspondence_distance=0.05,
        criteria=o3d.t.pipelines.registration.RANSACConvergenceCriteria(10000),
    )
    print(result)

    pcd0.transform(result.transformation)
    o3d.visualization.draw([pcd0.to_legacy(), pcd1.to_legacy()])
