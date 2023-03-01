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

    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    fragment_file_names = load_fragment_file_names(config)

    # Legacy
    pcd0 = o3d.io.read_point_cloud(fragment_file_names[0])
    pcd1 = o3d.io.read_point_cloud(fragment_file_names[5])
    fpfh0 = o3d.pipelines.registration.compute_fpfh_feature(
        pcd0, o3d.geometry.KDTreeSearchParamHybrid(0.1, 100)
    )
    fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(
        pcd1, o3d.geometry.KDTreeSearchParamHybrid(0.1, 100)
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd0, pcd1, fpfh0, fpfh1, False, 0.05,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False), 3,
        [
            o3d.pipelines.registration.
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                0.05)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            100000, 0.999))
    print(result)
    o3d.visualization.draw([pcd0.transform(result.transformation), pcd1])


    # Tensor

    for device in [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')]:
        pcd0 = o3d.t.io.read_point_cloud(fragment_file_names[0]).to(device)
        pcd1 = o3d.t.io.read_point_cloud(fragment_file_names[5]).to(device)
        fpfh0 = o3d.t.pipelines.registration.compute_fpfh_feature(pcd0)
        fpfh1 = o3d.t.pipelines.registration.compute_fpfh_feature(pcd1)

        print("start")
        result = o3d.t.pipelines.registration.ransac_from_features(
            pcd0,
            pcd1,
            fpfh0,
            fpfh1,
            max_correspondence_distance=0.05,
            criteria=o3d.t.pipelines.registration.RANSACConvergenceCriteria(100000),
        )
        print(result)

        pcd0.transform(result.transformation)
        o3d.visualization.draw([pcd0.to_legacy(), pcd1.to_legacy()])
