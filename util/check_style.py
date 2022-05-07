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

import subprocess
import re
import shutil
import argparse
from pathlib import Path
import multiprocessing
from functools import partial
import time
import sys

PYTHON_FORMAT_DIRS = [
    "examples",
    "docs",
    "python",
    "util",
]

JUPYTER_FORMAT_DIRS = [
    "examples",
]

# Note: also modify CPP_FORMAT_DIRS in check_cpp_style.cmake.
CPP_FORMAT_DIRS = [
    "cpp",
    "examples",
    "docs/_static",
]

# Yapf requires python 3.6+
if sys.version_info < (3, 6):
    raise RuntimeError("Requires Python 3.6+, currently using Python "
                       f"{sys.version_info.major}.{sys.version_info.minor}.")

# Check and import yapf
# > not found: throw exception
# > version mismatch: throw exception
try:
    import yapf
except ImportError:
    raise ImportError(
        "yapf not found. Install with `pip install yapf==0.30.0`.")
if yapf.__version__ != "0.30.0":
    raise RuntimeError(
        "yapf 0.30.0 required. Install with `pip install yapf==0.30.0`.")

# Check and import nbformat
# > not found: throw exception
try:
    import nbformat
except ImportError:
    raise ImportError(
        "nbformat not found. Install with `pip install nbformat`.")


class CppFormatter:

    standard_header = """// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
"""

    def __init__(self, file_paths, clang_format_bin):
        self.file_paths = file_paths
        self.clang_format_bin = clang_format_bin

    @staticmethod
    def _check_style(file_path, clang_format_bin):
        """
        Returns (true, true) if (style, header) is valid.
        """

        with open(file_path, 'r') as f:
            if f.read().startswith(CppFormatter.standard_header):
                is_valid_header = True
            else:
                is_valid_header = False

        cmd = [
            clang_format_bin,
            "-style=file",
            "-output-replacements-xml",
            file_path,
        ]
        result = subprocess.check_output(cmd).decode("utf-8")
        if "<replacement " in result:
            is_valid_style = False
        else:
            is_valid_style = True
        return (is_valid_style, is_valid_header)

    @staticmethod
    def _apply_style(file_path, clang_format_bin):
        cmd = [
            clang_format_bin,
            "-style=file",
            "-i",
            file_path,
        ]
        subprocess.check_output(cmd)

    def run(self, apply, no_parallel, verbose):
        num_procs = multiprocessing.cpu_count() if not no_parallel else 1
        action_name = "Applying C++/CUDA style" if apply else "Checking C++/CUDA style"
        print(f"{action_name} ({num_procs} process{'es'[:2*num_procs^2]})")
        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print(f"> {file_path}")

        start_time = time.time()
        with multiprocessing.Pool(num_procs) as pool:
            is_valid_files = pool.map(
                partial(self._check_style,
                        clang_format_bin=self.clang_format_bin),
                self.file_paths)

        changed_files = []
        wrong_header_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            is_valid_style = is_valid[0]
            is_valid_header = is_valid[1]
            if not is_valid_style:
                changed_files.append(file_path)
                if apply:
                    self._apply_style(file_path, self.clang_format_bin)
            if not is_valid_header:
                wrong_header_files.append(file_path)
        print(f"{action_name} took {time.time() - start_time:.2f}s")

        return (changed_files, wrong_header_files)


class PythonFormatter:

    standard_header = """# ----------------------------------------------------------------------------
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
"""

    def __init__(self, file_paths, style_config):
        self.file_paths = file_paths
        self.style_config = style_config

    @staticmethod
    def _check_style(file_path, style_config):
        """
        Returns (true, true) if (style, header) is valid.
        """

        with open(file_path, 'r') as f:
            content = f.read()
            if len(content) == 0 or content.startswith(
                    PythonFormatter.standard_header):
                is_valid_header = True
            else:
                is_valid_header = False

        _, _, changed = yapf.yapflib.yapf_api.FormatFile(
            file_path, style_config=style_config, in_place=False)
        return (not changed, is_valid_header)

    @staticmethod
    def _apply_style(file_path, style_config):
        _, _, _ = yapf.yapflib.yapf_api.FormatFile(file_path,
                                                   style_config=style_config,
                                                   in_place=True)

    def run(self, apply, no_parallel, verbose):
        num_procs = multiprocessing.cpu_count() if not no_parallel else 1
        action_name = "Applying Python style" if apply else "Checking Python style"
        print(f"{action_name} ({num_procs} process{'es'[:2*num_procs^2]})")

        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print(f"> {file_path}")

        start_time = time.time()
        with multiprocessing.Pool(num_procs) as pool:
            is_valid_files = pool.map(
                partial(self._check_style, style_config=self.style_config),
                self.file_paths)

        changed_files = []
        wrong_header_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            is_valid_style = is_valid[0]
            is_valid_header = is_valid[1]
            if not is_valid_style:
                changed_files.append(file_path)
                if apply:
                    self._apply_style(file_path, self.style_config)
            if not is_valid_header:
                wrong_header_files.append(file_path)

        print(f"{action_name} took {time.time() - start_time:.2f}s")
        return (changed_files, wrong_header_files)


class JupyterFormatter:

    def __init__(self, file_paths, style_config):
        self.file_paths = file_paths
        self.style_config = style_config

    @staticmethod
    def _check_or_apply_style(file_path, style_config, apply):
        """
        Returns true if style is valid.

        Since there are common code for check and apply style, the two functions
        are merged into one.
        """
        # Ref: https://gist.github.com/oskopek/496c0d96c79fb6a13692657b39d7c709
        with open(file_path, "r") as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)
        nbformat.validate(notebook)

        changed = False
        for cell in notebook.cells:
            if cell["cell_type"] != "code":
                continue
            src = cell["source"]
            lines = src.split("\n")
            if len(lines) <= 0 or "# noqa" in lines[0]:
                continue
            # yapf will puts a `\n` at the end of each cell, and if this is the
            # only change, cell_changed is still False.
            formatted_src, cell_changed = yapf.yapflib.yapf_api.FormatCode(
                src, style_config=style_config)
            if formatted_src.endswith("\n"):
                formatted_src = formatted_src[:-1]
            if cell_changed:
                cell["source"] = formatted_src
                changed = True

        if apply:
            with open(file_path, "w") as f:
                nbformat.write(notebook, f, version=nbformat.NO_CONVERT)

        return not changed

    def run(self, apply, no_parallel, verbose):
        num_procs = multiprocessing.cpu_count() if not no_parallel else 1
        action_name = "Applying Jupyter style" if apply else "Checking Jupyter style"
        print(f"{action_name} ({num_procs} process{'es'[:2*num_procs^2]})")

        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print(f"> {file_path}")

        start_time = time.time()
        with multiprocessing.Pool(num_procs) as pool:
            is_valid_files = pool.map(
                partial(self._check_or_apply_style,
                        style_config=self.style_config,
                        apply=False), self.file_paths)

        changed_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            if not is_valid:
                changed_files.append(file_path)
                if apply:
                    self._check_or_apply_style(file_path,
                                               style_config=self.style_config,
                                               apply=True)
        print(f"{action_name} took {time.time() - start_time:.2f}s")

        return changed_files


def _glob_files(directories, extensions):
    """
    Find files with certain extensions in directories recursively.

    Args:
        directories: list of directories, relative to the root Open3D repo directory.
        extensions: list of extensions, e.g. ["cpp", "h"].

    Return:
        List of file paths.
    """
    pwd = Path(__file__).resolve().parent
    open3d_root_dir = pwd.parent

    file_paths = []
    for directory in directories:
        directory = open3d_root_dir / directory
        for extension in extensions:
            extension_regex = "*." + extension
            file_paths.extend(directory.rglob(extension_regex))
    file_paths = [str(file_path) for file_path in file_paths]
    file_paths = sorted(list(set(file_paths)))
    return file_paths


def _find_clang_format():
    """
    Returns (bin_path, version) to clang-format version 10, throws exception
    otherwise.
    """
    required_clang_format_major = 10

    def parse_version(bin_path):
        """
        Get clang-format version string. Returns None if parsing fails.
        """
        version_str = subprocess.check_output([bin_path, "--version"
                                              ]).decode("utf-8").strip()
        match = re.match("^clang-format version ([0-9.]*).*$", version_str)
        return match.group(1) if match else None

    def parse_version_major(bin_path):
        """
        Get clang-format major version. Returns None if parsing fails.
        """
        version = parse_version(bin_path)
        return int(version.split(".")[0]) if version else None

    def find_bin_by_name(bin_name):
        """
        Returns bin path if found. Otherwise, returns None.
        """
        bin_path = shutil.which(bin_name)
        if bin_path is None:
            return None
        else:
            major = parse_version_major(bin_path)
            return bin_path if major == required_clang_format_major else None

    bin_path = find_bin_by_name("clang-format")
    if bin_path is not None:
        bin_version = parse_version(bin_path)
        return bin_path, bin_version

    bin_path = find_bin_by_name(f"clang-format-{required_clang_format_major}")
    if bin_path is not None:
        bin_version = parse_version(bin_path)
        return bin_path, bin_version

    raise RuntimeError(
        "clang-format not found. "
        "See http://www.open3d.org/docs/release/contribute/styleguide.html#style-guide "
        "for help on clang-format installation.")


def _filter_files(files, ignored_patterns):
    return [
        file for file in files if not any(
            [ignored_pattern in file for ignored_pattern in ignored_patterns])
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        default=False,
        help="Apply style to files in-place.",
    )
    parser.add_argument(
        "--no_parallel",
        dest="no_parallel",
        action="store_true",
        default=False,
        help="Disable parallel execution.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="If true, prints file names while formatting.",
    )
    args = parser.parse_args()

    # Check formatting libs
    clang_format_bin, clang_format_version = _find_clang_format()
    print(f"Using clang-format {clang_format_bin} ({clang_format_bin})")
    print(f"Using yapf {yapf.__version__} ({yapf.__file__})")
    print(f"Using nbformat {nbformat.__version__} ({nbformat.__file__})")
    pwd = Path(__file__).resolve().parent
    python_style_config = str(pwd.parent / ".style.yapf")

    cpp_ignored_files = ['cpp/open3d/visualization/shader/Shader.h']
    cpp_files = _glob_files(CPP_FORMAT_DIRS,
                            ["h", "cpp", "cuh", "cu", "isph", "ispc", "h.in"])
    cpp_files = _filter_files(cpp_files, cpp_ignored_files)

    # Check or apply style
    cpp_formatter = CppFormatter(cpp_files, clang_format_bin=clang_format_bin)
    python_formatter = PythonFormatter(_glob_files(PYTHON_FORMAT_DIRS, ["py"]),
                                       style_config=python_style_config)
    jupyter_formatter = JupyterFormatter(_glob_files(JUPYTER_FORMAT_DIRS,
                                                     ["ipynb"]),
                                         style_config=python_style_config)

    changed_files = []
    wrong_header_files = []
    changed_files_cpp, wrong_header_files_cpp = cpp_formatter.run(
        apply=args.apply, no_parallel=args.no_parallel, verbose=args.verbose)
    changed_files.extend(changed_files_cpp)
    wrong_header_files.extend(wrong_header_files_cpp)

    changed_files_python, wrong_header_files_python = python_formatter.run(
        apply=args.apply, no_parallel=args.no_parallel, verbose=args.verbose)
    changed_files.extend(changed_files_python)
    wrong_header_files.extend(wrong_header_files_python)

    changed_files.extend(
        jupyter_formatter.run(apply=args.apply,
                              no_parallel=args.no_parallel,
                              verbose=args.verbose))

    if len(changed_files) == 0 and len(wrong_header_files) == 0:
        print("All files passed style check")
        exit(0)

    if args.apply:
        if len(changed_files) != 0:
            print("Style applied to the following files:")
            print("\n".join(changed_files))
        if len(wrong_header_files) != 0:
            print("Please correct license header *manually* in the following "
                  "files (see util/check_style.py for the standard header):")
            print("\n".join(wrong_header_files))
            exit(1)
    else:
        error_files_no_duplicates = list(set(changed_files +
                                             wrong_header_files))
        if len(error_files_no_duplicates) != 0:
            print("Style error found in the following files:")
            print("\n".join(error_files_no_duplicates))
            exit(1)


if __name__ == "__main__":
    main()
