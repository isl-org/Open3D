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
    """Class decorator to register methodss with @register into a set."""
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


@registerableCLI
class Open3DMain:

    def __init__(self, test_mode: bool = False):
        self.banner = f"\n{'*' * 55}"
        self.banner += f"\n**                       Open3D                      **"
        self.banner += f"\n**      A Modern Library for 3D Data Processing      **\n{'*' * 55}"
        print(self.banner)

        print(self._get_friend_links())

        parser = argparse.ArgumentParser(description="Open3D CLI",
                                         usage=self._usage())
        parser.add_argument('command',
                            help="command from the above list to run")

        # Flag for unit testing
        self.test_mode = test_mode

        self.main_parser = parser

    @timer
    def __call__(self):
        # Print help if no command provided
        if len(sys.argv[1:2]) == 0:
            self.main_parser.print_help()
            return 1

        # Parse the command
        args = self.main_parser.parse_args(sys.argv[1:2])

        if args.command not in self.registered_commands:  # pylint: disable=E1101
            # TODO: do we really need this?
            if args.command.endswith(".py"):
                Open3DMain._exec_python_file(args.command)
            else:
                print(f"{args.command} is not a valid command!")
                self.main_parser.print_help()
            return 1

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
        msg = "\n"
        space = 20
        for command in sorted(self.registered_commands):  # pylint: disable=E1101
            msg += f"    {command}{' ' * (space - len(command))}|-> {getattr(self, command).__doc__}\n"
        return msg

    @staticmethod
    def _exec_python_file(filename: str):
        """Execute a Python file based on filename."""
        subprocess.call([sys.executable, filename] + sys.argv[1:])

    @staticmethod
    def _get_examples_dir() -> Path:
        """Get the path to the examples directory."""
        tools_path = os.path.dirname(os.path.abspath(__file__))
        examples_path = os.path.join(os.path.dirname(tools_path), "examples")
        examples_dir = Path(examples_path)
        return examples_dir

    @staticmethod
    def _get_available_examples() -> set:
        """Get a set of all available example names."""
        examples_dir = Open3DMain._get_examples_dir()
        all_examples = examples_dir.rglob('*.py')
        all_example_names = {f.stem: f.parent for f in all_examples}
        return all_example_names

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
        """Run an example by name (or name.py)"""
        choices = Open3DMain._get_available_examples()

        parser = argparse.ArgumentParser(prog='open3d example',
                                         description=f"{self.example.__doc__}")
        parser.add_argument(
            "name",
            help="Name of an example (supports .py extension too)\n",
            type=Open3DMain._example_choices_type(choices.keys()),
            choices=sorted(choices.keys()))
        parser.add_argument(
            '-p',
            '--print',
            required=False,
            dest='print',
            action='store_true',
            help="Print example source code instead of running it")
        parser.add_argument(
            '-P',
            '--pretty-print',
            required=False,
            dest='pretty_print',
            action='store_true',
            help="Like --print, but print in a rich format with line numbers")
        parser.add_argument(
            '-s',
            '--save',
            required=False,
            dest='save',
            action='store_true',
            help="Save source code to current directory instead of running it")

        # TODO: Pass the arguments to downstream correctly.
        args = parser.parse_args(arguments)

        examples_dir = Open3DMain._get_examples_dir()
        target = str(
            (examples_dir / choices[args.name] / f"{args.name}.py").resolve())
        # path for examples needs to be modified for implicit relative imports
        sys.path.append(str((examples_dir / choices[args.name]).resolve()))

        # Short circuit for testing
        if self.test_mode:
            return args

        if args.save:
            print(f"Saving example {args.name} to current directory...")
            shutil.copy(target, '.')
            return 0

        if args.pretty_print:
            try:
                import rich.console  # pylint: disable=C0415
                import rich.syntax  # pylint: disable=C0415
            except ImportError:
                print('To make -P work, please: python3 -m pip install rich')
                return 1
            # https://rich.readthedocs.io/en/latest/syntax.html
            syntax = rich.syntax.Syntax.from_path(target, line_numbers=True)
            console = rich.console.Console()
            console.print(syntax)
            return 0

        if args.print:
            with open(target) as f:
                print(f.read())
            return 0

        print(f"Running example {args.name} ...")

        runpy.run_path(target, run_name='__main__')

        return None

    @register
    def draw(self, arguments: list = sys.argv[2:]):
        """Visualize a mesh or pointcloud from a file"""
        parser = argparse.ArgumentParser(prog='open3d draw',
                                         description=f"{self.example.__doc__}")

        visgui = importlib.import_module("open3d.examples.gui.vis-gui")
        sys.argv = ['vis-gui.py', sys.argv[2]]
        visgui.main()

        # TODO: Use same function to execute examples and visualization script

        return None


def main():
    cli = Open3DMain()
    return cli()


if __name__ == "__main__":
    sys.exit(main())
