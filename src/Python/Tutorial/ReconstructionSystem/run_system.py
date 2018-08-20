import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction system")
    parser.add_argument("config", help="path to the config file")
    args = parser.parse_args()

    # check folder structure
    if args.config is not None:
        config = json.load(open(args.config))
        path_depth = os.path.join(config["path_dataset"], "depth")
        path_image = os.path.join(config["path_dataset"], "image")
        assert os.path.exists(path_depth), \
                "Path {} is not exist!".format(path_depth)
        assert os.path.exists(path_image), \
                "Path {} is not exist!".format(path_image)
    assert config is not None

    if config["run_make_fragments"]:
        import make_fragments
        make_fragments.run(config)
    if config["run_register_fragments"]:
        import register_fragments
        register_fragments.run(config)
    if config["run_integrate_scene"]:
        import integrate_scene
        integrate_scene.run(config)
