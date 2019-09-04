# Option 1: Do not define "PYTHON_EXECUTABLE", but run `cmake ..` within your
#           virtual environment. CMake will pick up the python executable in the
#           virtual environment.
# Option 2: You can also define `cmake -DPYTHON_EXECUTABLE` to specify a python
#           executable.
if (NOT PYTHON_EXECUTABLE)
    # find_program will returns the python executable in current PATH, which
    # works with virtualenv
    find_program(PYTHON_IN_PATH "python")
    set(PYTHON_EXECUTABLE ${PYTHON_IN_PATH})
endif()
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")

# Detect whether `npm` is installed. Jupyter support will only be enabled if
# `npm` is found.
find_program(NPM "npm")
if (ENABLE_JUPYTER)
    if (NOT NPM)
        message(WARNING "Cannot find npm. Jupyter support will be disabled.")
        set(ENABLE_JUPYTER OFF)
    else()
        message(STATUS "npm found at: ${NPM}. Jupyter support will be enabled.")
    endif()
endif()
message(STATUS "ENABLE_JUPYTER is set to ${ENABLE_JUPYTER}")

# We need to get python version to configure some meta files
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('%d.%d' % (sys.version_info.major, sys.version_info.minor))"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using Python version: ${PYTHON_VERSION}")

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('%d' % (sys.version_info.major,))"
    OUTPUT_VARIABLE PYTHON_VERSION_MAJOR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using Python version major: ${PYTHON_VERSION_MAJOR}")

set(PACKAGE_NAME open3d)

file(GLOB_RECURSE PY_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE PY_ALL_SOURCE_FILES "*.cpp")

if (NOT BUILD_AZURE_KINECT)
    list(REMOVE_ITEM PY_ALL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/io/sensor.cpp")
endif()

ADD_SOURCE_GROUP(Core)
ADD_SOURCE_GROUP(IO)
ADD_SOURCE_GROUP(Tutorial)
ADD_SOURCE_GROUP(Visualization)
pybind11_add_module(${PACKAGE_NAME}
    ${PY_ALL_HEADER_FILES} ${PY_ALL_SOURCE_FILES}
)

# Suppress Pybind11 warnings
target_include_directories(${PACKAGE_NAME} SYSTEM PRIVATE
    ${PYBIND11_INCLUDE_DIR}
)

# The Python headers won't be included outside the project, so it is okay to
# use the target_compile_definitions approach. Otherwise, use configure_file.
if (BUILD_AZURE_KINECT)
    target_compile_definitions(${PACKAGE_NAME} PRIVATE BUILD_AZURE_KINECT)
endif()

target_link_libraries(${PACKAGE_NAME} PRIVATE ${CMAKE_PROJECT_NAME})
if (${PYTHON_VERSION_MAJOR} EQUAL 2)
    target_compile_definitions(${PACKAGE_NAME} PRIVATE PYTHON_2_FALLBACK)
endif ()

# At `make`: open3d.so (or the equivalents) will be created at
# PYTHON_COMPILED_MODULE_DIR. The default locaiton is `build/lib/Python`
set(PYTHON_COMPILED_MODULE_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Python")
set_target_properties(${PACKAGE_NAME} PROPERTIES
                      FOLDER "Python"
                      LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
                      ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")

# Use `make python-package` to create the python package in the build directory
# The python package will be created at PYTHON_PACKAGE_DIR. It contains:
# 1) Pure-python code and misc files, copied from src/Python/package
# 2) The compiled python-C++ module, i.e. open3d.so (or the equivalents)
# 3) Configured files and supporting files
# Note: `make python-package` clears PYTHON_COMPILED_MODULE_DIR first every time
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
message(STATUS "PYPI_PACKAGE_NAME: ${PYPI_PACKAGE_NAME}")

add_custom_target(python-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPYTHON_VERSION=${PYTHON_VERSION}
            -DPYTHON_COMPILED_MODULE_PATH=$<TARGET_FILE:${PACKAGE_NAME}>
            -DENABLE_JUPYTER=${ENABLE_JUPYTER}
            -DPROJECT_EMAIL=${PROJECT_EMAIL}
            -DPROJECT_HOME=${PROJECT_HOME}
            -DPROJECT_DOCS=${PROJECT_DOCS}
            -DPROJECT_CODE=${PROJECT_CODE}
            -DPROJECT_ISSUES=${PROJECT_ISSUES}
            -DPROJECT_VERSION=${PROJECT_VERSION}
            -DPROJECT_VERSION_THREE_NUMBER=${PROJECT_VERSION_THREE_NUMBER}
            -DPYPI_PACKAGE_NAME=${PYPI_PACKAGE_NAME}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake
)

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)

# Use `make install-pip-package` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
    DEPENDS pip-package
)

# Use `make conda-package` to create conda package in the build directory
# Note that we don't provide `make install-conda-package` similar to pip. This
# is becuase:
#     1) `make install-pip-whell` works in conda environment for local build
#     2) `make conda-package` is mainly for internal use to distribute conda
add_custom_target(conda-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_and_install_conda_deps.cmake
    # If we put the following `conda-build` in check_and_install_conda_deps.cmake, it
    # causes broken pipe problem while running conda build. So we put it here.
    COMMAND conda-build conda_meta --output-folder conda_package
    COMMAND echo "conda package created at ${PYTHON_PACKAGE_DST_DIR}/conda_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)
