# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import argparse
import os
import runpy
import sys
from pathlib import Path
import open3d


class _Open3DArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        self.exit(2)


def _get_examples_dict():
    examples_dict = {
        "camera": ["camera_trajectory",],
        "geometry": [
            "image_processing",
            "kd_tree_feature_matching",
            "kd_tree_search",
            "octree_find_leaf",
            "octree_from_voxel_grid",
            "octree_point_cloud",
            "octree_traversal",
            "point_cloud_bounding_box",
            "point_cloud_convex_hull",
            "point_cloud_crop",
            "point_cloud_dbscan_clustering",
            "point_cloud_distance",
            "point_cloud_hidden_point_removal",
            "point_cloud_iss_keypoint_detector",
            "point_cloud_normal_estimation",
            "point_cloud_outlier_removal_radius",
            "point_cloud_outlier_removal_statistical",
            "point_cloud_paint",
            "point_cloud_plane_segmentation",
            "point_cloud_to_depth",
            "point_cloud_to_rgbd",
            "point_cloud_transformation",
            "point_cloud_voxel_downsampling",
            "point_cloud_with_numpy",
            "ray_casting_closest_geometry",
            "ray_casting_sdf",
            "ray_casting_to_image",
            "rgbd_datasets",
            "triangle_mesh_connected_components",
            "triangle_mesh_cropping",
            "triangle_mesh_deformation",
            "triangle_mesh_filtering_average",
            "triangle_mesh_from_point_cloud_alpha_shapes",
            "triangle_mesh_from_point_cloud_ball_pivoting",
            "triangle_mesh_from_point_cloud_poisson",
            "triangle_mesh_normal_estimation",
            "triangle_mesh_properties",
            "triangle_mesh_sampling",
            "triangle_mesh_simplification_decimation",
            "triangle_mesh_simplification_vertex_clustering",
            "triangle_mesh_subdivision",
            "triangle_mesh_transformation",
            "triangle_mesh_with_numpy",
            "voxel_grid_carving",
            "voxel_grid_from_point_cloud",
            "voxel_grid_from_triangle_mesh",
        ],
        "io": [
            "image_io",
            "point_cloud_io",
            "realsense_io",
            "triangle_mesh_io",
        ],
        "pipelines": [
            "colored_pointcloud_registration",
            "icp_registration",
            "multiway_registration",
            "pose_graph_optimization",
            "registration_fgr",
            "registration_ransac",
            "rgbd_integration_uniform",
            "rgbd_odometry",
            "robust_icp",
        ],
        "utility": ["vector",],
        "visualization": [
            "add_geometry",
            "all_widgets",
            "customized_visualization_key_action",
            "customized_visualization",
            "demo_scene",
            "draw_webrtc",
            "draw",
            "headless_rendering",
            "interactive_visualization",
            "line_width",
            "load_save_viewpoint",
            "mouse_and_point_coord",
            "multiple_windows",
            "non_blocking_visualization",
            "non_english",
            "online_processing",
            "remove_geometry",
            "render_to_image",
            "tensorboard_pytorch",
            "tensorboard_tensorflow",
            "text3d",
            "textured_mesh",
            "textured_model",
            "video",
            "vis_gui",
        ]
    }

    return examples_dict


def _get_examples_dir():
    """Get the path to the examples directory."""
    tools_path = os.path.dirname(os.path.abspath(__file__))
    examples_path = os.path.join(os.path.dirname(tools_path), "examples")
    examples_dir = Path(examples_path)
    return examples_dir


def _get_example_categories():
    """Get a set of all available category names."""
    examples_dict = _get_examples_dict()
    all_categories = [category for category in examples_dict]
    return all_categories


def _get_examples_in_category(category):
    """Get a set of example names in given cateogry."""
    examples_dict = _get_examples_dict()
    examples_dir = _get_examples_dir()
    category_path = os.path.join(examples_dir, category)
    example_names = {
        name: Path(category_path) for name in examples_dict[category]
    }
    return example_names


def _support_choice_with_dot_py(choice):
    if choice.endswith(".py"):
        return choice[:-3]
    return choice


def _example_help_categories():
    msg = f"\ncategories:\n"
    for category in sorted(_get_example_categories()):
        msg += f"  {category}\n"
    msg += ("\nTo view the example in each category run:\n"
            "  open3d example --list category\n ")
    return msg


def _example(parser, args):

    if args.category_example == None:
        if args.list:
            print(_example_help_categories())
        else:
            parser.print_help()
        return 0

    try:
        category = args.category_example.split("/")[0]
        example = args.category_example.split("/")[1]
    except:
        category = args.category_example
        example = ""

    if category not in _get_example_categories():
        print("error: invalid category provided: " + category)
        parser.print_help()
        parser.exit(2)

    if args.list or example == "":
        print("examples in " + category + ": ")
        for examples_in_category in sorted(_get_examples_in_category(category)):
            print("  " + str(examples_in_category))
        print("\nTo view all categories run:")
        print("  open3d example --list\n")
        return 0

    examples_dir = _get_examples_dir()
    examples_in_category = _get_examples_in_category(category)
    target = str((examples_dir / category / examples_in_category[example] /
                  f"{example}.py").resolve())
    # path for examples needs to be modified for implicit relative imports
    sys.path.append(
        (examples_dir / category / examples_in_category[example]).resolve())

    if args.show:
        with open(target) as f:
            print(f.read())
        return 0

    print(f"Running example {args.category_example} ...")
    removed_args = sys.argv[1:3]
    del sys.argv[1:3]
    runpy.run_path(target, run_name="__main__")
    sys.argv.insert(1, removed_args[0])
    sys.argv.insert(2, removed_args[1])

    return 0


def _draw(parser, args):
    if args.filename == None:
        parser.print_help()

    removed_arg = sys.argv[1]
    sys.argv.pop(1)
    import open3d.app as app
    app.main()
    sys.argv.insert(1, removed_arg)

    return 0


def main():
    print(f"*******************************************************\n"
          f"** Open3D: A Modern Library for 3D Data Processing   **\n"
          f"**                                                   **\n"
          f"** Version {open3d.__version__: <25}                 **\n"
          f"** Docs    http://www.open3d.org/docs                **\n"
          f"** Code    https://github.com/isl-org/Open3D         **\n"
          f"*******************************************************")

    main_parser = _Open3DArgumentParser(
        description="Open3D commad-line tools",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter)
    main_parser.add_argument("-V",
                             "--version",
                             action="version",
                             version="Open3D " + open3d.__version__,
                             help="Show program's version number and exit.")
    main_parser.add_argument("-h",
                             "--help",
                             action="help",
                             help="Show this help message and exit.")

    subparsers = main_parser.add_subparsers(
        title="command",
        dest="command",
        description='Functionalities supported by Open3D CLI',
        help="Select one of these commands\n ")

    example_help = (
        "Run an Open3D example. Example usage: \n"
        "```bash\n"
        "open3d example --list                                  # List examples\n"
        "open3d example --list geometry                         # List examples in geometry\n"
        "open3d example geometry/point_cloud_convex_hull        # Run an example\n"
        "open3d example --show geometry/point_cloud_convex_hull # Show source code of an example\n"
        "```\n")
    parser_example = subparsers.add_parser(
        "example",
        add_help=False,
        description=example_help + _example_help_categories(),
        help=example_help,
        formatter_class=argparse.RawTextHelpFormatter)
    parser_example.add_argument(
        "category_example",
        nargs="?",
        help=
        "Category/example_name of an example (supports .py extension too)\n",
        type=_support_choice_with_dot_py)
    parser_example.add_argument("example_args",
                                nargs="*",
                                help="Arguments for the example to be run\n")
    parser_example.add_argument(
        "-l",
        "--list",
        required=False,
        dest="list",
        action="store_true",
        help="List all categories or examples available\n"
        "usage       :  open3d example --list \n"
        "               open3d example --list category\n"
        "for example :  open3d example --list geometry\n ")
    parser_example.add_argument(
        "-s",
        "--show",
        required=False,
        dest="show",
        action="store_true",
        help="Show example source code instead of running it\n"
        "usage       :  open3d example --show category/example_name\n"
        "for example :  open3d example --show geometry/triangle_mesh_deformation\n "
    )
    parser_example.add_argument("-h",
                                "--help",
                                action="help",
                                help="Show this help message and exit.")
    parser_example.set_defaults(func=_example)

    draw_help = ("Visualize a mesh or pointcloud from a file\n"
                 "for example :  open3d draw")
    parser_draw = subparsers.add_parser(
        "draw",
        description=draw_help,
        help=draw_help,
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter)

    parser_draw.add_argument("filename",
                             nargs="?",
                             help="Name of the mesh or point cloud file.")
    parser_draw.add_argument("-h",
                             "--help",
                             action="help",
                             help="Show this help message and exit.")
    parser_draw.set_defaults(func=_draw)

    args = main_parser.parse_args()
    if args.command in subparsers.choices.keys():
        return args.func(subparsers.choices[args.command], args)
    else:
        main_parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
