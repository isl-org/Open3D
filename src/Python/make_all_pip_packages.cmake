# Warning: Internal use only, consider droping this in the future
# Use `make all-pip-pacakges` to create the pip package in the build directory
# This creates: open3d-python, py3d, open3d-original, open3d-official, open-3d
#               pip wheels

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
configure_file("${PYTHON_PACKAGE_SRC_DIR}/MANIFEST.in"
               "${PYTHON_PACKAGE_DST_DIR}/MANIFEST.in" COPYONLY)
configure_file("${PYTHON_PACKAGE_SRC_DIR}/open3d/__init__.py.in"
               "${PYTHON_PACKAGE_DST_DIR}/open3d/__init__.py")

# We distributes multiple PyPI pacakges under different names
set(PYPI_PACKAGE_NAMES
    open3d-python
    py3d
    open3d-original
    open3d-official
    open-3d
)
foreach (PYPI_PACKAGE_NAME ${PYPI_PACKAGE_NAMES})
    message("Making PyPI package: ${PYPI_PACKAGE_NAME}...")
    configure_file("${PYTHON_PACKAGE_SRC_DIR}/setup.py.in"
                   "${PYTHON_PACKAGE_DST_DIR}/setup.py")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
)
endforeach()
