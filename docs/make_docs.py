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

# Sphinx makefile with api docs generation
# (1) The user call `make *` (e.g. `make html`) gets forwarded to make.py
# (2) make.py generate Python api docs, one ".rst" file per class / function
# (3) make.py calls the actual `sphinx-build`

import argparse
import subprocess
import sys
import importlib
import os
import inspect
import shutil
import re
from pathlib import Path
import nbformat
import nbconvert
import ssl
import certifi
import urllib.request


def _create_or_clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print("Removed directory %s" % dir_path)
    os.makedirs(dir_path)
    print("Created directory %s" % dir_path)


def _update_file(src, dst):
    """Copies a file if the destination does not exist or is older."""
    if Path(dst).exists():
        src_stat = os.stat(src)
        dst_stat = os.stat(dst)
        if src_stat.st_mtime - dst_stat.st_mtime <= 0:
            print("Copy skipped: {}".format(dst))
            return
    print("Copy: {}\n   -> {}".format(src, dst))
    shutil.copy2(src, dst)


class PyAPIDocsBuilder:
    """
    Generate Python API *.rst files, per (sub) module, per class, per function.
    The file name is the full module name.

    E.g. If output_dir == "python_api", the following files are generated:
    python_api/open3d.camera.rst
    python_api/open3d.camera.PinholeCameraIntrinsic.rst
    ...
    """

    def __init__(self, output_dir="python_api", input_dir="python_api_in"):
        """
        input_dir: The input dir for custom rst files that override the
                   generated files.
        """
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.module_names = PyAPIDocsBuilder._get_documented_module_names()
        print("Generating *.rst Python API docs in directory: %s" %
              self.output_dir)

    def generate_rst(self):
        _create_or_clear_dir(self.output_dir)

        for module_name in self.module_names:
            try:
                module = self._try_import_module(module_name)
                self._generate_module_class_function_docs(module_name, module)
            except:
                print("[Warning] Module {} cannot be imported.".format(
                    module_name))

    @staticmethod
    def _get_documented_module_names():
        """Reads the modules of the python api from the index.rst"""
        module_names = []
        with open("documented_modules.txt", "r") as f:
            for line in f:
                print(line, end="")
                m = re.match("^(open3d\..*)\s*$", line)
                if m:
                    module_names.append(m.group(1))
        print("Documented modules:")
        for module_name in module_names:
            print("-", module_name)
        return module_names

    def _try_import_module(self, full_module_name):
        """Returns the module object for the given module path"""
        import open3d  # make sure the root module is loaded
        if open3d._build_config['BUILD_TENSORFLOW_OPS']:
            import open3d.ml.tf
        if open3d._build_config['BUILD_PYTORCH_OPS']:
            import open3d.ml.torch

        try:
            # Try to import directly. This will work for pure python submodules
            module = importlib.import_module(full_module_name)
            return module
        except ImportError:
            # Traverse the module hierarchy of the root module.
            # This code path is necessary for modules for which we manually
            # define a specific module path (e.g. the modules defined with
            # pybind).
            current_module = open3d
            for sub_module_name in full_module_name.split(".")[1:]:
                current_module = getattr(current_module, sub_module_name)
            return current_module

    def _generate_function_doc(self, full_module_name, function_name,
                               output_path):
        out_string = ""
        out_string += "%s.%s" % (full_module_name, function_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name
        out_string += "\n\n" + ".. autofunction:: %s" % function_name
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    def _generate_class_doc(self, full_module_name, class_name, output_path):
        out_string = ""
        out_string += "%s.%s" % (full_module_name, class_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name
        out_string += "\n\n" + ".. autoclass:: %s" % class_name
        out_string += "\n    :members:"
        out_string += "\n    :undoc-members:"
        if not (full_module_name.startswith("open3d.ml.tf") or
                full_module_name.startswith("open3d.ml.torch")):
            out_string += "\n    :inherited-members:"
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    def _generate_module_doc(self, full_module_name, class_names,
                             function_names, sub_module_names,
                             sub_module_doc_path):
        class_names = sorted(class_names)
        function_names = sorted(function_names)
        out_string = ""
        out_string += full_module_name
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name

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

        if len(sub_module_names) > 0:
            out_string += "\n\n**Modules**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for sub_module_name in sub_module_names:
                out_string += "\n    " + "%s" % (sub_module_name,)
            out_string += "\n"

        obj_names = class_names + function_names + sub_module_names
        if len(obj_names) > 0:
            out_string += "\n\n.. toctree::"
            out_string += "\n    :hidden:"
            out_string += "\n"
            for obj_name in obj_names:
                out_string += "\n    %s <%s.%s>" % (
                    obj_name,
                    full_module_name,
                    obj_name,
                )
            out_string += "\n"

        with open(sub_module_doc_path, "w") as f:
            f.write(out_string)

    def _generate_module_class_function_docs(self, full_module_name, module):
        print("Generating docs for submodule: %s" % full_module_name)

        # Class docs
        class_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.isclass(obj[1]) and not obj[0].startswith('_')
        ]
        for class_name in class_names:
            file_name = "%s.%s.rst" % (full_module_name, class_name)
            output_path = os.path.join(self.output_dir, file_name)
            input_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(input_path):
                shutil.copyfile(input_path, output_path)
                continue
            self._generate_class_doc(full_module_name, class_name, output_path)

        # Function docs
        function_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.isroutine(obj[1]) and not obj[0].startswith('_')
        ]
        for function_name in function_names:
            file_name = "%s.%s.rst" % (full_module_name, function_name)
            output_path = os.path.join(self.output_dir, file_name)
            input_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(input_path):
                shutil.copyfile(input_path, output_path)
                continue
            self._generate_function_doc(full_module_name, function_name,
                                        output_path)

        # Submodule docs, only supports 2-level nesting
        # (i.e. open3d.pipeline.registration)
        sub_module_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.ismodule(obj[1]) and not obj[0].startswith('_')
        ]
        documented_sub_module_names = [
            sub_module_name for sub_module_name in sub_module_names if "%s.%s" %
            (full_module_name, sub_module_name) in self.module_names
        ]

        # Path
        sub_module_doc_path = os.path.join(self.output_dir,
                                           full_module_name + ".rst")
        input_path = os.path.join(self.input_dir, full_module_name + ".rst")
        if os.path.isfile(input_path):
            shutil.copyfile(input_path, sub_module_doc_path)
            return
        self._generate_module_doc(
            full_module_name,
            class_names,
            function_names,
            documented_sub_module_names,
            sub_module_doc_path,
        )


class PyExampleDocsBuilder:
    """
    Generate Python examples *.rst files.
    """

    def __init__(self, input_dir, pwd, output_dir="python_example"):
        self.output_dir = Path(str(output_dir))
        self.input_dir = Path(str(input_dir))
        self.prefixes = [
            ("image", "Image"),
            ("kd_tree", "KD Tree"),
            ("octree", "Octree"),
            ("point_cloud", "Point Cloud"),
            ("ray_casting", "Ray Casting"),
            ("rgbd", "RGBD Image"),
            ("triangle_mesh", "Triangle Mesh"),
            ("voxel_grid", "Voxel Grid"),
        ]

        sys.path.append(os.path.join(pwd, "..", "python", "tools"))
        from cli import _get_all_examples_dict
        self.get_all_examples_dict = _get_all_examples_dict
        print("Generating *.rst Python example docs in directory: %s" %
              self.output_dir)

    def _get_examples_dict(self):
        examples_dict = self.get_all_examples_dict()
        categories_to_remove = [
            "benchmark", "reconstruction_system", "t_reconstruction_system"
        ]
        for cat in categories_to_remove:
            examples_dict.pop(cat)
        return examples_dict

    def _get_prefix(self, example_name):
        for prefix, sub_category in self.prefixes:
            if example_name.startswith(prefix):
                return prefix
        raise Exception("No prefix found for geometry examples")

    @staticmethod
    def _generate_index(title, output_path):
        os.makedirs(output_path)
        out_string = (f"{title}\n" f"{'-' * len(title)}\n\n")
        with open(output_path / "index.rst", "w") as f:
            f.write(out_string)

    @staticmethod
    def _add_example_to_docs(example, output_path):
        shutil.copy(example, output_path)
        out_string = (f"{example.stem}.py"
                      f"\n```````````````````````````````````````\n"
                      f"\n.. literalinclude:: {example.stem}.py"
                      f"\n   :language: python"
                      f"\n   :linenos:"
                      f"\n\n\n")

        with open(output_path / "index.rst", "a") as f:
            f.write(out_string)

    def generate_rst(self):
        _create_or_clear_dir(self.output_dir)
        examples_dict = self._get_examples_dict()

        categories = [cat for cat in self.input_dir.iterdir() if cat.is_dir()]

        for cat in categories:
            if cat.stem in examples_dict.keys():
                out_dir = self.output_dir / cat.stem
                if (cat.stem == "geometry"):
                    self._generate_index(cat.stem.capitalize(), out_dir)
                    with open(out_dir / "index.rst", "a") as f:
                        f.write(f".. toctree::\n" f"    :maxdepth: 2\n\n")
                        for prefix, sub_cat in self.prefixes:
                            f.write(f"    {prefix}/index\n")

                    for prefix, sub_category in self.prefixes:
                        self._generate_index(sub_category, out_dir / prefix)
                    examples = sorted(Path(cat).glob("*.py"))
                    for ex in examples:
                        if ex.stem in examples_dict[cat.stem]:
                            prefix = self._get_prefix(ex.stem)
                            sub_category_path = out_dir / prefix
                            self._add_example_to_docs(ex, sub_category_path)
                else:
                    if (cat.stem == "io"):
                        self._generate_index("IO", out_dir)
                    else:
                        self._generate_index(cat.stem.capitalize(), out_dir)

                    examples = sorted(Path(cat).glob("*.py"))
                    for ex in examples:
                        if ex.stem in examples_dict[cat.stem]:
                            shutil.copy(ex, out_dir)
                            self._add_example_to_docs(ex, out_dir)


class SphinxDocsBuilder:
    """
    SphinxDocsBuilder calls Python api and examples docs generation and then
    calls sphinx-build:

    (1) The user call `make *` (e.g. `make html`) gets forwarded to make.py
    (2) Calls PyAPIDocsBuilder to generate Python api docs rst files
    (3) Calls `sphinx-build` with the user argument
    """

    def __init__(self, current_file_dir, html_output_dir, is_release,
                 skip_notebooks):
        self.current_file_dir = current_file_dir
        self.html_output_dir = html_output_dir
        self.is_release = is_release
        self.skip_notebooks = skip_notebooks

    def run(self):
        """
        Call Sphinx command with hard-coded "html" target
        """
        # Copy docs files from Open3D-ML repo
        open3d_ml_root = os.environ.get(
            "OPEN3D_ML_ROOT",
            os.path.join(self.current_file_dir, "../../Open3D-ML"))
        open3d_ml_docs = [
            os.path.join(open3d_ml_root, "docs", "tensorboard.md")
        ]
        for open3d_ml_doc in open3d_ml_docs:
            if os.path.isfile(open3d_ml_doc):
                shutil.copy(open3d_ml_doc, self.current_file_dir)

        build_dir = os.path.join(self.html_output_dir, "html")

        if self.is_release:
            version_list = [
                line.rstrip("\n").split(" ")[1]
                for line in open("../cpp/open3d/version.txt")
            ]
            release_version = ".".join(version_list[:3])
            print("Building docs for release:", release_version)

            cmd = [
                "sphinx-build",
                "-b",
                "html",
                "-D",
                "version=" + release_version,
                "-D",
                "release=" + release_version,
                ".",
                build_dir,
            ]
        else:
            cmd = [
                "sphinx-build",
                "-b",
                "html",
                ".",
                build_dir,
            ]

        sphinx_env = os.environ.copy()
        sphinx_env[
            "skip_notebooks"] = "true" if self.skip_notebooks else "false"

        print('Calling: "%s"' % " ".join(cmd))
        print('Env: "%s"' % sphinx_env)
        subprocess.check_call(cmd,
                              env=sphinx_env,
                              stdout=sys.stdout,
                              stderr=sys.stderr)


class DoxygenDocsBuilder:

    def __init__(self, html_output_dir):
        self.html_output_dir = html_output_dir

    def run(self):
        doxygen_temp_dir = "doxygen"
        _create_or_clear_dir(doxygen_temp_dir)

        cmd = ["doxygen", "Doxyfile"]
        print('Calling: "%s"' % " ".join(cmd))
        subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        shutil.copytree(
            os.path.join("doxygen", "html"),
            os.path.join(self.html_output_dir, "html", "cpp_api"),
        )

        if os.path.exists(doxygen_temp_dir):
            shutil.rmtree(doxygen_temp_dir)


class JupyterDocsBuilder:

    def __init__(self, current_file_dir, clean_notebooks, execute_notebooks):
        self.clean_notebooks = clean_notebooks
        self.execute_notebooks = execute_notebooks
        self.current_file_dir = current_file_dir
        print("Notebook execution mode: {}".format(self.execute_notebooks))

    def overwrite_tutorial_file(self, url, output_file, output_file_path):
        with urllib.request.urlopen(
                url,
                context=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            with open(output_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        shutil.move(output_file, output_file_path)

    def run(self):
        if self.execute_notebooks == "never":
            return

        # Setting os.environ["CI"] will disable interactive (blocking) mode in
        # Jupyter notebooks
        os.environ["CI"] = "true"

        # Copy test_data directory to the tutorial folder
        test_data_in_dir = (Path(self.current_file_dir).parent / "examples" /
                            "test_data")
        test_data_out_dir = Path(self.current_file_dir) / "test_data"
        if test_data_out_dir.exists():
            shutil.rmtree(test_data_out_dir)
        shutil.copytree(test_data_in_dir, test_data_out_dir)

        # Copy and execute notebooks in the tutorial folder
        nb_paths = []
        nb_direct_copy = [
            'tensor.ipynb', 'hashmap.ipynb', 't_icp_registration.ipynb',
            'jupyter_visualization.ipynb'
        ]
        example_dirs = [
            "geometry", "core", "pipelines", "visualization", "t_pipelines"
        ]
        for example_dir in example_dirs:
            in_dir = (Path(self.current_file_dir) / "jupyter" / example_dir)
            out_dir = Path(self.current_file_dir) / "tutorial" / example_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                in_dir.parent / "open3d_tutorial.py",
                out_dir.parent / "open3d_tutorial.py",
            )

            if self.clean_notebooks:
                for nb_out_path in out_dir.glob("*.ipynb"):
                    print("Delete: {}".format(nb_out_path))
                    nb_out_path.unlink()

            for nb_in_path in in_dir.glob("*.ipynb"):
                nb_out_path = out_dir / nb_in_path.name
                _update_file(nb_in_path, nb_out_path)
                nb_paths.append(nb_out_path)

            # Copy the 'images' dir present in some example dirs.
            if (in_dir / "images").is_dir():
                if (out_dir / "images").exists():
                    shutil.rmtree(out_dir / "images")
                print("Copy: {}\n   -> {}".format(in_dir / "images",
                                                  out_dir / "images"))
                shutil.copytree(in_dir / "images", out_dir / "images")

        # Execute Jupyter notebooks
        for nb_path in nb_paths:
            if nb_path.name in nb_direct_copy:
                print("[Processing notebook {}, directly copied]".format(
                    nb_path.name))
                continue

            print("[Processing notebook {}]".format(nb_path.name))
            with open(nb_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # https://github.com/spatialaudio/nbsphinx/blob/master/src/nbsphinx.py
            has_code = any(c.source for c in nb.cells if c.cell_type == "code")
            has_output = any(
                c.get("outputs") or c.get("execution_count")
                for c in nb.cells
                if c.cell_type == "code")
            execute = (self.execute_notebooks == "auto" and has_code and
                       not has_output) or self.execute_notebooks == "always"
            print("has_code: {}, has_output: {}, execute: {}".format(
                has_code, has_output, execute))

            if execute:
                ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=6000)
                try:
                    ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
                except nbconvert.preprocessors.execute.CellExecutionError:
                    print("Execution of {} failed, this will cause CI to fail.".
                          format(nb_path.name))
                    if "GITHUB_ACTIONS" in os.environ:
                        raise

                with open(nb_path, "w", encoding="utf-8") as f:
                    nbformat.write(nb, f)

        url = "https://github.com/isl-org/Open3D/files/7592880/t_icp_registration.zip"
        output_file = "t_icp_registration.ipynb"
        output_file_path = os.path.join(
            self.current_file_dir,
            "tutorial/t_pipelines/t_icp_registration.ipynb")
        self.overwrite_tutorial_file(url, output_file, output_file_path)


if __name__ == "__main__":
    """
    # Clean existing notebooks in docs/tutorial, execute notebooks if the
    # notebook does not have outputs, and build docs for Python and C++.
    $ python make_docs.py --clean_notebooks --execute_notebooks=auto --sphinx --doxygen

    # Build docs for Python (--sphinx) and C++ (--doxygen).
    $ python make_docs.py --execute_notebooks=auto --sphinx --doxygen

    # Build docs for release (version number will be used instead of git hash).
    $ python make_docs.py --is_release --sphinx --doxygen
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--clean_notebooks",
        action="store_true",
        default=False,
        help=("Whether to clean existing notebooks in docs/tutorial. "
              "Notebooks are copied from examples/python to docs/tutorial."),
    )
    parser.add_argument(
        "--execute_notebooks",
        default="auto",
        choices=("auto", "always", "never"),
        help="Jupyter notebook execution mode.",
    )
    parser.add_argument(
        "--py_api_rst",
        default="always",
        choices=("always", "never"),
        help="Build Python API documentation in reST format.",
    )
    parser.add_argument(
        "--py_example_rst",
        default="always",
        choices=("always", "never"),
        help="Build Python example documentation in reST format.",
    )
    parser.add_argument(
        "--sphinx",
        action="store_true",
        default=False,
        help="Build Sphinx for main docs and Python API docs.",
    )
    parser.add_argument(
        "--doxygen",
        action="store_true",
        default=False,
        help="Build Doxygen for C++ API docs.",
    )
    parser.add_argument(
        "--is_release",
        action="store_true",
        default=False,
        help="Show Open3D version number rather than git hash.",
    )
    args = parser.parse_args()

    pwd = os.path.dirname(os.path.realpath(__file__))

    # Clear output dir if new docs are to be built
    html_output_dir = os.path.join(pwd, "_out")
    _create_or_clear_dir(html_output_dir)

    # Clear C++ build directory
    cpp_build_dir = os.path.join(pwd, "_static", "C++", "build")
    if os.path.exists(cpp_build_dir):
        shutil.rmtree(cpp_build_dir)
        print("Removed directory %s" % cpp_build_dir)

    # Python API reST docs
    if not args.py_api_rst == "never":
        print("Building Python API reST")
        pd = PyAPIDocsBuilder()
        pd.generate_rst()

    # Python example reST docs
    py_example_input_dir = os.path.join(pwd, "..", "examples", "python")
    if not args.py_example_rst == "never":
        print("Building Python example reST")
        pe = PyExampleDocsBuilder(input_dir=py_example_input_dir, pwd=pwd)
        pe.generate_rst()

    # Jupyter docs (needs execution)
    if not args.execute_notebooks == "never":
        print("Building Jupyter docs")
        jdb = JupyterDocsBuilder(pwd, args.clean_notebooks,
                                 args.execute_notebooks)
        jdb.run()

    # Sphinx is hard-coded to build with the "html" option
    # To customize build, run sphinx-build manually
    if args.sphinx:
        print("Building Sphinx docs")
        skip_notebooks = (args.execute_notebooks == "never" and
                          args.clean_notebooks)
        sdb = SphinxDocsBuilder(pwd, html_output_dir, args.is_release,
                                skip_notebooks)
        sdb.run()
    else:
        print("Sphinx build disabled, use --sphinx to enable")

    # Doxygen is hard-coded to build with default option
    # To customize build, customize Doxyfile or run doxygen manually
    if args.doxygen:
        print("Doxygen build enabled")
        ddb = DoxygenDocsBuilder(html_output_dir)
        ddb.run()
    else:
        print("Doxygen build disabled, use --doxygen to enable")
