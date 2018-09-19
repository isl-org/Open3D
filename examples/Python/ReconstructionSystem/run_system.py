import os
import sys
import json
import argparse
sys.path.append("../Utility")
from file import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction system")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--make",
            help="Step 1) make fragments from RGBD sequence",
            action="store_true")
    parser.add_argument("--register",
            help="Step 2) register all fragments to detect loop closure",
            action="store_true")
    parser.add_argument("--refine",
            help="Step 3) refine rough registrations", action="store_true")
    parser.add_argument("--integrate",
            help="Step 4) integrate the whole RGBD sequence to make final mesh",
            action="store_true")
    parser.add_argument("--debug_mode", help="turn on debug mode",
            action="store_true")
    args = parser.parse_args()

    if not args.make and \
            not args.register and \
            not args.refine and \
            not args.integrate:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # check folder structure
    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
            check_folder_structure(config["path_dataset"])
    assert config is not None

    if args.debug_mode:
        config['debug_mode'] = True
    else:
        config['debug_mode'] = False

    # set default parameters if not specified
    if not config['n_frames_per_fragment']:
        config['n_frames_per_fragment'] = 100
    if not config['n_keyframes_per_n_frame']:
        config['n_keyframes_per_n_frame'] = 5
    if not config['max_depth']:
        config['max_depth'] = 3.0
    if not config['voxel_size']:
        config['voxel_size'] = 0.05
    if not config['max_depth_diff']:
        config['max_depth_diff'] = 0.07
    if not config['preference_loop_closure_odometry']:
        config['preference_loop_closure_odometry'] = 0.1
    if not config['preference_loop_closure_registration']:
        config['preference_loop_closure_registration'] = 5.0
    if not config['tsdf_cubic_size']:
        config['tsdf_cubic_size'] = 3.0
    if not config['icp_method']:
        config['icp_method'] = "color"
    if not config['global_registration']:
        config['global_registration'] = "ransac",
    if not config['python_multi_threading']
        config['python_multi_threading'] = True

    if args.make:
        import make_fragments
        make_fragments.run(config)
    if args.register:
        import register_fragments
        register_fragments.run(config)
    if args.refine:
        import refine_registration
        refine_registration.run(config)
    if args.integrate:
        import integrate_scene
        integrate_scene.run(config)
