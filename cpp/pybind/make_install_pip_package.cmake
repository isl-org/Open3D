# We need this file for cross-platform support on Windows
# For Ubuntu/Mac, we can simply do `pip install ${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl -U`

# Note: Since `make python-package` clears PYTHON_COMPILED_MODULE_DIR every time,
#       it is guaranteed that there is only one wheel in ${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl
file(GLOB WHEEL_FILE "${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl")
execute_process(COMMAND ${Python3_EXECUTABLE} -m pip uninstall open3d --yes)
execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install ${WHEEL_FILE} -U)
