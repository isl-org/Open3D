# We need this file for cross-platform support on Windows
# For Ubuntu/Mac, we can simply do `pip install ${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl -U`

# Note: Since `make python-package` clears PYTHON_COMPILED_MODULE_DIR every time,
#       it is guaranteed that there is only one wheel in ${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl
file(GLOB WHEEL_FILE "${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl")
execute_process(
    COMMAND pip install --upgrade ${WHEEL_FILE}
    # `pip install --upgrade` sometimes does not install the wheel for
    # uncommited changes. This is because the Open3D package version number
    # `open3d-version_id+commit_id` remains the same.
    COMMAND pip install --upgrade --no-deps --force-reinstall ${WHEEL_FILE}
)
