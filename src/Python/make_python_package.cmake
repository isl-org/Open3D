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

# Create python pacakge. It contains
# 1) Pure-python code and misc files, copied from src/Python/Package
# 2) The compiled python-C++ module, i.e. open3d.so (or the equivalents)
# 3) Configured files and supporting files

# Clean up directory
file(REMOVE_RECURSE ${PYTHON_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/open3d)

# 1) Pure-python code and misc files, copied from src/Python/Package
file(COPY ${PYTHON_PACKAGE_SRC_DIR}/
     DESTINATION ${PYTHON_PACKAGE_DST_DIR}
     PATTERN "*.in" EXCLUDE
)

# 2) The compiled python-C++ module, i.e. open3d.so (or the equivalents)
get_filename_component(PYTHON_COMPILED_MODULE_NAME ${PYTHON_COMPILED_MODULE_PATH} NAME)
file(COPY ${PYTHON_COMPILED_MODULE_PATH}
     DESTINATION ${PYTHON_PACKAGE_DST_DIR}/open3d)

# 3) Configured files and supporting files
configure_file("${PYTHON_PACKAGE_SRC_DIR}/setup.py.in"
               "${PYTHON_PACKAGE_DST_DIR}/setup.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/MANIFEST.in"
               "${PYTHON_PACKAGE_DST_DIR}/MANIFEST.in" COPYONLY)
configure_file("${PYTHON_PACKAGE_SRC_DIR}/open3d/__init__.py.in"
               "${PYTHON_PACKAGE_DST_DIR}/open3d/__init__.py")
