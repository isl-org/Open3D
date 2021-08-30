from pathlib import Path
import os
import subprocess
import re
import os
import shutil
from pathlib import Path
import functools

pwd = Path(os.path.dirname(os.path.realpath(__file__)))

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


@functools.cache  # Python 3.9+
def find_clang_format():
    """
    Find clang-format:
      - not found: throw exception
      - version mismatch: print warning
    """
    preferred_clang_format_name = "clang-format-10"
    preferred_version_major = 10
    clang_format_bin = shutil.which(preferred_clang_format_name)
    if clang_format_bin is None:
        clang_format_bin = shutil.which("clang-format")
    if clang_format_bin is None:
        raise RuntimeError(
            "clang-format not found. "
            "See http://www.open3d.org/docs/release/contribute/styleguide.html#style-guide "
            "for help on clang-format installation.")
    version_str = subprocess.check_output([clang_format_bin, "--version"
                                          ]).decode("utf-8").strip()
    try:
        m = re.match("^clang-format version ([0-9.]*).*$", version_str)
        if m:
            version_str = m.group(1)
            version_str_token = version_str.split(".")
            major = int(version_str_token[0])
            if major != preferred_version_major:
                print("Warning: {} required, but got {}.".format(
                    preferred_clang_format_name, version_str))
        else:
            raise
    except:
        print("Warning: failed to parse clang-format version {}, "
              "please ensure {} is used.".format(version_str,
                                                 preferred_clang_format_name))
    print("Using clang-format version {}.".format(version_str))

    return clang_format_bin


def apply_style(file_path):
    clang_format_bin = find_clang_format()
    cmd = [
        clang_format_bin,
        "-style=file",
        "-i",
        file_path,
    ]
    subprocess.check_output(cmd)


def glob_files(extensions):
    files = []
    for extension in extensions:
        files += list(pwd.glob('**/*.' + extension))
    return files


def is_using_tensor_assert(content):
    if any(pattern in content
           for pattern in ["AssertShape", "AssertDevice", "AssertDtype"]):
        return True
    return False


def is_using_new_tensor_assert(content):
    if any(pattern in content for pattern in
           ["AssertTensorShape", "AssertTensorDevice", "AssertTensorDtype"]):
        return True
    return False


for path in glob_files(["cpp", "h", "hpp", "cu", "cuh"]):
    relative_path = path.relative_to(pwd)

    if str(relative_path).startswith("build"):
        continue

    if str(relative_path).startswith("3rdparty"):
        continue

    if any(skip_file in str(relative_path) for skip_file in
           ["Shader.h", "Tensor.h", "Tensor.cpp", "TensorList", "TensorCheck"]):
        continue

    with open(path, 'r') as f:
        content = f.read()
        if content[:len(standard_header)] != standard_header:
            raise RuntimeError(
                f"{relative_path} is not following the standard header")

    if False and is_using_tensor_assert(content):
        if '#include "open3d/utility/Logging.h"' in content:
            new_content = content
            new_content = new_content.replace(
                '#include "open3d/utility/Logging.h"',
                '#include "open3d/utility/Logging.h"\n#include "open3d/core/TensorCheck.h"'
            )
        elif "#pragma once" in content:
            new_content = content
            new_content = new_content.replace(
                "#pragma once",
                '#pragma once\n\n#include "open3d/core/TensorCheck.h"')
        elif '#include "open3d/Open3D.h"' in content:
            new_content = content
        else:
            new_content = standard_header
            new_content += "\n\n"
            new_content += '#include "open3d/core/TensorCheck.h"'
            new_content += "\n"
            new_content += content[len(standard_header):]

        with open(path, "w") as f:
            f.write(new_content)

        apply_style(path)
        print(f"{relative_path} updated")

    if True and is_using_new_tensor_assert(content):
        if "namespace core" in content:
            new_content = content
            new_content = new_content.replace("core::AssertTensorShape",
                                              "AssertTensorShape")
            new_content = new_content.replace("core::AssertTensorDevice",
                                              "AssertTensorDevice")
            new_content = new_content.replace("core::AssertTensorDtype",
                                              "AssertTensorDtype")

            with open(path, "w") as f:
                f.write(new_content)

            apply_style(path)
            print(f"{relative_path} updated")
