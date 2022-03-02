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
from nis import cat
import os
import runpy
import sys
from pathlib import Path
from unicodedata import category

import open3d as o3d
import open3d.app as app


class _Open3DArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        sys.stderr.write("Error: %s\n" % message)
        self.print_help()
        self.exit(2)


def _get_examples_dir():
    """Get the path to the examples directory."""
    tools_path = os.path.dirname(os.path.abspath(__file__))
    examples_path = os.path.join(os.path.dirname(tools_path), "examples")
    if os.path.exists(examples_path):
        examples_dir = Path(examples_path)
        return examples_dir
    else:
        examples_path = os.path.join(
            os.path.dirname(os.path.dirname(tools_path)), "examples", "python")
        examples_dir = Path(examples_path)
        return examples_dir


def _get_all_examples_dict():
    ex_dir = _get_examples_dir()
    categories = [cat for cat in ex_dir.iterdir() if cat.is_dir()]
    examples_dict = {}
    for cat_path in categories:
        examples = sorted(Path(cat_path).glob("*.py"))
        if len(examples) > 0:
            examples_dict[cat_path.stem] = [
                ex.stem for ex in examples if ex.stem != "__init__"
            ]
    return examples_dict


def _get_runnable_examples_dict():
    examples_dict = _get_all_examples_dict()
    categories_to_remove = [
        "benchmark", "reconstruction_system", "t_reconstruction_system"
    ]
    examples_to_remove = {
        "io": ["realsense_io",],
        "visualization": [
            "online_processing",
            "tensorboard_pytorch",
            "tensorboard_tensorflow",
        ]
    }
    for cat in categories_to_remove:
        examples_dict.pop(cat)
    for cat in examples_to_remove.keys():
        for ex in examples_to_remove[cat]:
            examples_dict[cat].remove(ex)
    return examples_dict


def _get_all_examples():
    all_examples = []
    examples_dict = _get_runnable_examples_dict()
    for category in examples_dict:
        for example in examples_dict[category]:
            all_examples.append(f"{category}/{example}")
    return all_examples


def _get_example_categories():
    """Get a set of all available category names."""
    examples_dict = _get_runnable_examples_dict()
    all_categories = [category for category in examples_dict]
    return all_categories


def _get_examples_in_category(category):
    """Get a set of example names in given cateogry."""
    examples_dict = _get_runnable_examples_dict()
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
    msg += "\nTo view the example in each category, run one of the following commands:\n"
    for category in sorted(_get_example_categories()):
        msg += f"  open3d example --list {category}\n"
    return msg


def _example(parser, args):

    if args.category_example == None:
        if args.list:
            for category in _get_example_categories():
                print("examples in " + category + ": ")
                for examples_in_category in sorted(
                        _get_examples_in_category(category)):
                    print(f"  {category}/{examples_in_category}")
                print("")
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
        print("Error: invalid category provided: " + category)
        parser.print_help()
        parser.exit(2)

    if args.list:
        if example == "":
            print("examples in " + category + ": ")
            for examples_in_category in sorted(
                    _get_examples_in_category(category)):
                print(f"  {category}/{examples_in_category}")
            print("\nTo view all examples, run:")
            print("  open3d example --list\n")
            return 0
        else:
            print("Error: invalid category provided: " + args.category_example)
            parser.print_help()
            parser.exit(2)

    if args.category_example not in _get_all_examples():
        print("Error: invalid example name provided: " + args.category_example)
        parser.print_help()
        parser.exit(2)

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
    elif not os.path.isfile(args.filename):
        print(f"Error: could not find file: {args.filename}")
        parser.print_help()
        parser.exit(2)

    removed_arg = sys.argv[1]
    sys.argv.pop(1)
    app.main()
    sys.argv.insert(1, removed_arg)
    return 0


def main():
    print(f"***************************************************\n"
          f"* Open3D: A Modern Library for 3D Data Processing *\n"
          f"*                                                 *\n"
          f"* Version {o3d.__version__: <22}                  *\n"
          f"* Docs    http://www.open3d.org/docs              *\n"
          f"* Code    https://github.com/isl-org/Open3D       *\n"
          f"***************************************************")

    main_parser = _Open3DArgumentParser(
        description="Open3D commad-line tools",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter)
    main_parser.add_argument("-V",
                             "--version",
                             action="version",
                             version="Open3D " + o3d.__version__,
                             help="Show program's version number and exit.")
    main_parser.add_argument("-h",
                             "--help",
                             action="help",
                             help="Show this help message and exit.")

    subparsers = main_parser.add_subparsers(
        title="command",
        dest="command",
        help="Select one of these commands.\n ")

    example_help = (
        "View or run an Open3D example. Example usage: \n"
        "  open3d example --list                                  # List examples\n"
        "  open3d example --list geometry                         # List examples in geometry\n"
        "  open3d example geometry/point_cloud_convex_hull        # Run an example\n"
        "  open3d example --show geometry/point_cloud_convex_hull # Show source code of an example\n\n"
    )
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
        "usage:\n"
        "  open3d example --list \n"
        "  open3d example --list [category]\n"
        "e.g.:\n"
        "  open3d example --list geometry\n ")
    parser_example.add_argument(
        "-s",
        "--show",
        required=False,
        dest="show",
        action="store_true",
        help="Show example source code instead of running it\n"
        "usage:\n"
        "  open3d example --show [category]/[example_name]\n"
        "e.g.:\n"
        "  open3d example --show geometry/triangle_mesh_deformation\n ")
    parser_example.add_argument("-h",
                                "--help",
                                action="help",
                                help="Show this help message and exit.")
    parser_example.set_defaults(func=_example)

    draw_help = (
        "Load and visualize a 3D model. Example usage:\n"
        "  open3d draw                                            # Start a blank Open3D viewer\n"
        "  open3d draw path/to/model_file                         # Visualize a 3D model file\n"
    )
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
