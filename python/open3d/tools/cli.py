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
import math
import os
import runpy
import shutil
import subprocess
import sys
import timeit
from functools import wraps
from pathlib import Path
import importlib


def timer(func):
    """Function decorator to benchmark a function running time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start
        print(f">>> Running time: {elapsed:.2f}s")
        return result

    return wrapper


def registerableCLI(cls):
    """Class decorator to register methods with @register into a set."""
    cls.registered_commands = set([])
    for name in dir(cls):
        method = getattr(cls, name)
        if hasattr(method, 'registered'):
            cls.registered_commands.add(name)
    return cls


def register(func):
    """Method decorator to register CLI commands."""
    func.registered = True
    return func


class Open3DArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


class CommandHelp:

    def __init__(self, description, usage, example):
        self.description = description
        self.usage = usage
        self.example = example


@registerableCLI
class Open3DMain:

    def __init__(self):
        self.banner = f"\n{'*' * 55}"
        self.banner += f"\n**                       Open3D                      **"
        self.banner += f"\n**      A Modern Library for 3D Data Processing      **\n{'*' * 55}"
        print(self.banner)

        print(self._get_friend_links())

        self.command_helper = dict()
        for command in sorted(self.registered_commands):
            try:
                self.command_helper[command] = CommandHelp(
                    description=getattr(self, command).__doc__.split('|')[0],
                    usage=getattr(self, command).__doc__.split('|')[1],
                    example=getattr(self, command).__doc__.split('|')[2])
            except:
                print("Error: ", command,
                      getattr(self, command).__doc__.split('|'))
                exit()

        parser = Open3DArgumentParser(
            description="Open3D CLI",
            usage=self._usage(),
            add_help=False,
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('command',
                            help="command from the above list to run")
        parser.add_argument('-V',
                            '--version',
                            action='version',
                            version='Open3D v0.14.1',
                            help="Show program's version number and exit\n"
                            "usage : open3d --verison\n ")
        parser.add_argument('-h',
                            '--help',
                            action='help',
                            help='Show this help message and exit\n'
                            'usage : open3d --help\n ')

        self.main_parser = parser

    @timer
    def __call__(self):
        # Parse the command
        args = self.main_parser.parse_args(sys.argv[1:2])
        return getattr(self, args.command)(sys.argv[2:])

    @staticmethod
    def _get_friend_links():
        return '\n' \
               'Docs:     http://www.open3d.org/docs/release/\n' \
               'GitHub:   https://github.com/isl-org/Open3D\n' \
               'Discord:  https://discord.com/invite/D35BGvn\n'

    def _usage(self) -> str:
        """Compose deterministic usage message based on registered_commands."""
        # TODO: add some color to commands
        msg = '\n    open3d <command> <additional_arguments ...>\n\n' \
              'commands:\n'
        space = 15
        for command in sorted(self.registered_commands):
            msg += f"  {command}{' ' * (space - len(command))}{self.command_helper[command].description}\n"
            msg += f"  {' ' * space}usage       :  {self.command_helper[command].usage}\n"
            msg += f"  {' ' * space}for example :  {self.command_helper[command].example}\n\n"
        return msg

    @staticmethod
    def _exec_python_file(filename: str,):
        """Execute a Python file based on filename."""
        subprocess.call([sys.executable, filename] + sys.argv[2:])

    @staticmethod
    def _get_examples_dir() -> Path:
        """Get the path to the examples directory."""
        tools_path = os.path.dirname(os.path.abspath(__file__))
        examples_path = os.path.join(os.path.dirname(tools_path),
                                     "../../examples/python/")
        examples_dir = Path(examples_path)
        return examples_dir

    @staticmethod
    def _get_example_categories() -> list:
        """Get a set of all available category names."""
        examples_dir = Open3DMain._get_examples_dir()
        all_categories = [x.name for x in examples_dir.iterdir() if x.is_dir()]
        if "__pycache__" in all_categories:
            all_categories.remove("__pycache__")
        return all_categories

    @staticmethod
    def _get_available_examples() -> set:
        """Get a set of all available example names."""
        examples_dir = Open3DMain._get_examples_dir()
        all_examples = examples_dir.rglob('*.py')
        all_example_names = {f.stem: f.parent for f in all_examples}
        return all_example_names

    @staticmethod
    def _get_examples_in_category(category) -> set:
        """Get a set of example names in given cateogry."""
        examples_dir = Open3DMain._get_examples_dir()
        category_path = os.path.join(examples_dir, category)
        examples_in_category = Path(category_path).rglob('*.py')
        example_names = {f.stem: f.parent for f in examples_in_category}
        if "__init__" in example_names:
            example_names.pop("__init__")
        return example_names

    @staticmethod
    def _example_choices_type(choices):

        def support_choice_with_dot_py(choice):
            if choice.endswith('.py') and choice.split('.')[0] in choices:
                # try to find and remove python file extension
                return choice.split('.')[0]
            return choice

        return support_choice_with_dot_py

    @register
    def example(self, arguments: list = sys.argv[2:]):
        'Run an example by category/name (or category/name.py) |' \
        'open3d example <category>/<name> <example_args ...> |' \
        'open3d example geometry/mesh_deformation'

        def usage_long():
            # TODO: add some color to commands
            msg = self.command_helper["example"].usage
            msg += "\n\n"
            msg += "categories:\n"
            for category in sorted(self._get_example_categories()):
                msg += "  " + str(category) + "\n"
            msg += "\nTo view the example in each category run:\n"
            msg += "  open3d example --list category\n "
            return msg

        choices = Open3DMain._get_available_examples()

        parser = Open3DArgumentParser(
            prog='open3d example',
            description=self.command_helper["example"].description,
            usage=usage_long(),
            add_help=False,
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument(
            "name",
            nargs='?',
            help="Category/name of an example (supports .py extension too)\n",
            type=Open3DMain._example_choices_type(choices.keys()))
        parser.add_argument("example_args",
                            nargs='*',
                            help='Arguments for the example to be run\n')
        parser.add_argument('-l',
                            '--list',
                            required=False,
                            dest='list',
                            action='store_true',
                            help='List all categories or examples available\n'
                            'usage       :  open3d example --list \n'
                            '               open3d example --list <category>\n'
                            'for example :  open3d example --list geometry\n ')
        parser.add_argument(
            '-s',
            '--show',
            required=False,
            dest='show',
            action='store_true',
            help='Show example source code instead of running it\n'
            'usage       :  open3d example --show <category>/<example_name>\n'
            'for example :  open3d example --show geometry/mesh_deformation\n ')
        parser.add_argument(
            '-p',
            '--show_pretty',
            required=False,
            dest='show_pretty',
            action='store_true',
            help='Like --show, but show in a rich format with line numbers\n'
            'usage       :  open3d example --show_pretty <category>/<example_name>\n'
            'for example :  open3d example --show_pretty geometry/mesh_deformation\n '
        )
        parser.add_argument('-h',
                            '--help',
                            action='help',
                            help='Show this help message and exit\n'
                            'usage       :  open3d example --help\n ')

        args = parser.parse_args(arguments)

        if args.name == None:
            if args.list:
                print("categories: ")
                for category in sorted(self._get_example_categories()):
                    print("  " + str(category))
                print("\nTo view the example in each category run:")
                print("  open3d example --list <category>\n")
                return 0
            else:
                sys.stderr.write(
                    'error: the following arguments are required: name\n')
                parser.print_help()
                return 1

        try:
            category = args.name.split('/')[0]
            example = args.name.split('/')[1]
        except:
            category = args.name
            example = ""

        if category not in self._get_example_categories():
            print('error: invalid category provided: ' + category)
            parser.print_help()
            return 1

        if args.list:
            print("examples in " + category + ": ")
            for examples_in_category in sorted(
                    self._get_examples_in_category(category)):
                print("  " + str(examples_in_category))
            print("\nTo view all categories run:")
            print("  open3d example --list\n")
            return 0

        examples_dir = Open3DMain._get_examples_dir()
        examples_in_category = self._get_examples_in_category(category)
        target = str((examples_dir / category / examples_in_category[example] /
                      f"{example}.py").resolve())
        # path for examples needs to be modified for implicit relative imports
        sys.path.append(
            str((examples_dir / category /
                 examples_in_category[example]).resolve()))

        if args.show_pretty:
            try:
                import rich.console
                import rich.syntax
            except ImportError:
                print(
                    'To make -p work, please run: python3 -m pip install rich')
                return 1
            # https://rich.readthedocs.io/en/latest/syntax.html
            syntax = rich.syntax.Syntax.from_path(target, line_numbers=True)
            console = rich.console.Console()
            console.print(syntax)
            return 0

        if args.show:
            with open(target) as f:
                print(f.read())
            return 0

        print(f"Running example {args.name} ...")
        removed_args = sys.argv[1:3]
        del sys.argv[1:3]
        runpy.run_path(target, run_name='__main__')
        sys.argv.insert(1, removed_args[0])
        sys.argv.insert(2, removed_args[1])

        return None

    @register
    def draw(self, arguments: list = sys.argv[2:]):
        'Visualize a mesh or pointcloud from a file |' \
        'open3d draw <filename> |' \
        'open3d draw'

        parser = Open3DArgumentParser(
            prog='open3d draw',
            description=self.command_helper["draw"].description,
            usage=self.command_helper["draw"].usage,
            add_help=False,
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument("filename",
                            nargs='?',
                            help="Name of the mesh or point cloud file")
        parser.add_argument('-h',
                            '--help',
                            action='help',
                            help='Show this help message and exit\n'
                            'usage       :  open3d example --help\n ')

        args = parser.parse_args(arguments)

        if args.filename == None:
            parser.print_help()

        examples_dir = Open3DMain._get_examples_dir()
        visgui = str(examples_dir / "gui" / "vis-gui.py")
        # path for examples needs to be modified for implicit relative imports
        sys.path.append(str(examples_dir / "gui" / "vis-gui.py"))
        removed_arg = sys.argv[1]
        sys.argv.pop(1)
        runpy.run_path(visgui, run_name='__main__')
        sys.argv.insert(1, removed_arg)
        return None


def main():
    cli = Open3DMain()
    return cli()


if __name__ == "__main__":
    sys.exit(main())
