# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
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

# Sphinx makefile with api docs generation
# (1) The user call `make *` (e.g. `make html`) gets forwarded to make.py
# (2) make.py generate Python api docs, one ".rst" file per class / function
# (3) make.py calls the actual `sphinx-build`

from __future__ import print_function
import argparse
import subprocess
import sys
import multiprocessing
import importlib
import os
from inspect import getmembers, isbuiltin, isclass, ismodule
import shutil

# Global sphinx options
SPHINX_BUILD = "sphinx-build"
SOURCE_DIR = "."
BUILD_DIR = "_build"


def create_or_clear(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print("Removed directory %s" % dir_path)
    os.makedirs(dir_path)
    print("Created directory %s" % dir_path)


class PyDocsBuilder:
    """
    Generate Python API *.rst files, per (sub) module, per class, per function.
    The file name is the full module name.

    E.g. If output_dir == "python_api", the following files are generated:
    python_api/open3d.camera.rst
    python_api/open3d.camera.PinholeCameraIntrinsic.rst
    ...
    """

    def __init__(self, output_dir, c_module, c_module_relative):
        self.output_dir = output_dir
        self.c_module = c_module
        self.c_module_relative = c_module_relative
        print("Generating *.rst Python API docs in directory: %s" %
              self.output_dir)

    def generate_rst(self):
        create_or_clear(self.output_dir)

        main_c_module = importlib.import_module(self.c_module)
        sub_module_names = sorted(
            [obj[0] for obj in getmembers(main_c_module) if ismodule(obj[1])])
        for sub_module_name in sub_module_names:
            PyDocsBuilder._generate_sub_module_class_function_docs(
                sub_module_name, self.output_dir)

    @staticmethod
    def _generate_function_doc(sub_module_full_name, function_name,
                               output_path):
        # print("Generating docs: %s" % (output_path,))
        out_string = ""
        out_string += "%s.%s" % (sub_module_full_name, function_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % sub_module_full_name
        out_string += "\n\n" + ".. autofunction:: %s" % function_name
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    @staticmethod
    def _generate_class_doc(sub_module_full_name, class_name, output_path):
        # print("Generating docs: %s" % (output_path,))
        out_string = ""
        out_string += "%s.%s" % (sub_module_full_name, class_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % sub_module_full_name
        out_string += "\n\n" + ".. autoclass:: %s" % class_name
        out_string += "\n    :members:"
        out_string += "\n    :undoc-members:"
        out_string += "\n    :inherited-members:"
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    @staticmethod
    def _generate_sub_module_doc(sub_module_name, class_names, function_names,
                                 sub_module_doc_path):
        # print("Generating docs: %s" % (sub_module_doc_path,))
        class_names = sorted(class_names)
        function_names = sorted(function_names)
        sub_module_full_name = "open3d.%s" % (sub_module_name,)
        out_string = ""
        out_string += sub_module_full_name
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % sub_module_full_name

        if len(class_names) > 0:
            out_string += "\n\n**Classes**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for class_name in class_names:
                out_string += "\n    " + "%s" % (class_name,)
            out_string += "\n"

        if len(function_names) > 0:
            out_string += "\n\n**Functions**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for function_name in function_names:
                out_string += "\n    " + "%s" % (function_name,)
            out_string += "\n"

        obj_names = class_names + function_names
        if len(obj_names) > 0:
            out_string += "\n\n.. toctree::"
            out_string += "\n    :hidden:"
            out_string += "\n"
            for obj_name in obj_names:
                out_string += "\n    %s <%s.%s>" % (
                    obj_name,
                    sub_module_full_name,
                    obj_name,
                )
            out_string += "\n"

        with open(sub_module_doc_path, "w") as f:
            f.write(out_string)

    @staticmethod
    def _generate_sub_module_class_function_docs(sub_module_name, output_dir):
        sub_module = importlib.import_module("open3d.open3d.%s" %
                                             (sub_module_name,))
        sub_module_full_name = "open3d.%s" % (sub_module_name,)
        print("Generating docs for submodule: %s" % sub_module_full_name)

        # Class docs
        class_names = [
            obj[0] for obj in getmembers(sub_module) if isclass(obj[1])
        ]
        for class_name in class_names:
            file_name = "%s.%s.rst" % (sub_module_full_name, class_name)
            output_path = os.path.join(output_dir, file_name)
            PyDocsBuilder._generate_class_doc(sub_module_full_name, class_name,
                                              output_path)

        # Function docs
        function_names = [
            obj[0] for obj in getmembers(sub_module) if isbuiltin(obj[1])
        ]
        for function_name in function_names:
            file_name = "%s.%s.rst" % (sub_module_full_name, function_name)
            output_path = os.path.join(output_dir, file_name)
            PyDocsBuilder._generate_function_doc(sub_module_full_name,
                                                 function_name, output_path)

        # Submodule docs
        sub_module_doc_path = os.path.join(output_dir,
                                           sub_module_full_name + ".rst")
        PyDocsBuilder._generate_sub_module_doc(sub_module_name, class_names,
                                               function_names,
                                               sub_module_doc_path)


class SphinxDocsBuilder:
    """
    # SphinxDocsBuilder calls Python api docs generation and then calls
    # sphinx-build:
    #
    # (1) The user call `make *` (e.g. `make html`) gets forwarded to make.py
    # (2) Calls PyDocsBuilder to generate Python api docs rst files
    # (3) Calls `sphinx-build` with the user argument
    """

    valid_makefile_args = {
        "changes",
        "clean",
        "coverage",
        "devhelp",
        "dirhtml",
        "doctest",
        "epub",
        "gettext",
        "help",
        "html",
        "htmlhelp",
        "info",
        "json",
        "latex",
        "latexpdf",
        "latexpdfja",
        "linkcheck",
        "man",
        "pickle",
        "pseudoxml",
        "qthelp",
        "singlehtml",
        "texinfo",
        "text",
        "xml",
    }

    def __init__(self, makefile_arg):
        if makefile_arg not in self.valid_makefile_args:
            print('Invalid make argument: "%s", displaying help.' %
                  makefile_arg)
            self.is_valid_arg = False
        else:
            self.is_valid_arg = True
            self.makefile_arg = makefile_arg

        # Hard-coded parameters for Python API docs generation for now
        # Directory structure for the Open3D Python package:
        # open3d
        # - __init__.py
        # - open3d.so  # Actual name depends on OS and Python version
        self.c_module = "open3d.open3d"  # Points to the open3d.so
        self.c_module_relative = "open3d"  # The relative module reference to open3d.so
        self.python_api_output_dir = "python_api"

    def run(self):
        if not self.is_valid_arg:
            self.makefile_arg = "help"
        elif self.makefile_arg == "clean":
            print("Removing directory %s" % self.python_api_output_dir)
            shutil.rmtree(self.python_api_output_dir, ignore_errors=True)
        elif self.makefile_arg == "help":
            pass  # Do not call self._gen_python_api_docs()
        else:
            self._gen_python_api_docs()
        self._run_sphinx()

    def _gen_python_api_docs(self):
        """
        Generate Python docs.
        Each module, class and function gets one .rst file.
        """
        pd = PyDocsBuilder(self.python_api_output_dir, self.c_module,
                           self.c_module_relative)
        pd.generate_rst()

    def _run_sphinx(self):
        """
        Call Sphinx command with self.makefile_arg
        """
        create_or_clear(BUILD_DIR)
        cmd = [
            SPHINX_BUILD,
            "-M",
            self.makefile_arg,
            SOURCE_DIR,
            BUILD_DIR,
            "-j",
            str(multiprocessing.cpu_count()),
        ]
        print('Calling: "%s"' % " ".join(cmd))
        subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("makefile_arg", nargs="?")
    args = parser.parse_args()

    sdb = SphinxDocsBuilder(args.makefile_arg)
    sdb.run()
