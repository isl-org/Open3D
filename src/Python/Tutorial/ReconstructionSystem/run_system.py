import os
import sys
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction system")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--make_fragments",
            help="Step 1) making fragments from RGBD sequence", action="store_true")
    parser.add_argument("--register_fragments",
            help="Step 2) register fragments", action="store_true")
    parser.add_argument("--integrate_scene",
            help="Step 3) integrate the whole RGBD sequence", action="store_true")
    parser.add_argument("--debug_mode",
            help="turn on debug mode", action="store_true")
    args = parser.parse_args()

    if not args.make_fragments and \
            not args.register_fragments and \
            not args.integrate_scene:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # check folder structure
    if args.config is not None:
        config = json.load(open(args.config))
        path_depth = os.path.join(config["path_dataset"], "depth")
        path_image = os.path.join(config["path_dataset"], "image")
        assert os.path.exists(path_depth), \
                "Path %s is not exist!" % path_depth
        assert os.path.exists(path_image), \
                "Path %s is not exist!" % path_image
    assert config is not None

    if args.debug_mode:
        config['debug_mode'] = True
    else:
        config['debug_mode'] = False

    if args.make_fragments:
        import make_fragments
        make_fragments.run(config)
    if args.register_fragments:
        import register_fragments
        register_fragments.run(config)
    if args.integrate_scene:
        import integrate_scene
        integrate_scene.run(config)
