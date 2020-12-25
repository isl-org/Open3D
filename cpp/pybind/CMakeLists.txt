if (NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE should be set in top level CMakeLists.txt")
endif()

# Detect whether `npm` is installed. Jupyter support will only be enabled if
# `npm` is found.
find_program(NPM "npm")
if (BUILD_JUPYTER_EXTENSION)
    if (NOT NPM)
        message(WARNING "Cannot find npm. Jupyter support will be disabled.")
        set(BUILD_JUPYTER_EXTENSION OFF)
    else()
        message(STATUS "npm found at: ${NPM}. Jupyter support will be enabled.")
    endif()
endif()
message(STATUS "BUILD_JUPYTER_EXTENSION is set to ${BUILD_JUPYTER_EXTENSION}")

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

if (${PYTHON_VERSION_MAJOR} EQUAL 2)
    message(FATAL_ERROR "Python 2 is no longer supported. Please check your \
            your Python version with `python --version`. To use Python 3, do \
            one of the following: \
            (1) activate a Python 3 virtual environment; \
            (2) specify the Python path explicitly with -DPYTHON_EXECUTABLE=/path/to/python3; \
            (3) add Python 3 to the PATH environment variable.")
endif()

set(PACKAGE_NAME pybind)

file(GLOB_RECURSE PY_ALL_SOURCE_FILES "*.cpp")

if (NOT BUILD_GUI)
    list(REMOVE_ITEM PY_ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/visualization/gui/gui.cpp")
    list(REMOVE_ITEM PY_ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/visualization/rendering/rendering.cpp")
    list(REMOVE_ITEM PY_ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/visualization/o3dvisualizer.cpp")
endif()

if (NOT BUILD_AZURE_KINECT)
    list(REMOVE_ITEM PY_ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/io/sensor.cpp")
endif()

if (NOT BUILD_RPC_INTERFACE)
    list(REMOVE_ITEM PY_ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/io/rpc.cpp"
        )
endif()

# NO_EXTRAS disables LTO which causes problems during link with nvcc 9.x
if ( CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.0.0" )
    pybind11_add_module(${PACKAGE_NAME} NO_EXTRAS
        ${PY_ALL_HEADER_FILES} ${PY_ALL_SOURCE_FILES}
    )
else ()
    pybind11_add_module(${PACKAGE_NAME}
        ${PY_ALL_HEADER_FILES} ${PY_ALL_SOURCE_FILES}
    )
endif ()

# Include with `#include "pybind/geometry/xxx.h`
target_include_directories(${PACKAGE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Suppress Pybind11 warnings
target_include_directories(${PACKAGE_NAME} SYSTEM PRIVATE
    ${PYBIND11_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
open3d_set_global_properties(${PACKAGE_NAME})

# The Python headers won't be included outside the project, so it is okay to
# use the target_compile_definitions approach. Otherwise, use configure_file.
if (BUILD_AZURE_KINECT)
    target_compile_definitions(${PACKAGE_NAME} PRIVATE BUILD_AZURE_KINECT)
endif()
if (BUILD_LIBREALSENSE)
    target_compile_definitions(${PACKAGE_NAME} PRIVATE BUILD_LIBREALSENSE)
endif()
if (BUILD_GUI)
    target_compile_definitions(${PACKAGE_NAME} PRIVATE BUILD_GUI)
endif()
if (WITH_FAISS)
    target_compile_definitions(${PACKAGE_NAME} PRIVATE WITH_FAISS)
endif()

target_link_libraries(${PACKAGE_NAME} PRIVATE ${CMAKE_PROJECT_NAME})
if (WIN32)
    target_link_options(${PACKAGE_NAME} PUBLIC "/force:multiple")
endif()

# At `make`: open3d.so (or the equivalents) will be created at
# PYTHON_COMPILED_MODULE_DIR. The default location is
# `build/lib/${CMAKE_BUILD_TYPE}/Python/{cpu|cuda}`
set(PYTHON_COMPILED_MODULE_DIR
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Python/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
set_target_properties(${PACKAGE_NAME} PROPERTIES
                      FOLDER "Python"
                      LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
                      ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")
# Include additional libraries that may be absent from the user system
# eg: libc++.so and libc++abi.so (needed by filament) for Linux.
# libc++.so is a linker script including libc++.so.1 and libc++abi.so, so append 1 to libc++.so
set(PYTHON_EXTRA_LIBRARIES "")
if (BUILD_GUI AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND PYTHON_EXTRA_LIBRARIES ${CPP_LIBRARY}.1 ${CPPABI_LIBRARY})
endif()

# Use `make python-package` to create the python package in the build directory
# The python package will be created at PYTHON_PACKAGE_DIR. It contains:
# 1) Pure-python code and misc files, copied from python/package
# 2) The compiled python-C++ module, i.e. open3d.so (or the equivalents)
# 3) Configured files and supporting files
# Note: `make python-package` clears PYTHON_COMPILED_MODULE_DIR first every time
set(PYTHON_PACKAGE_SRC_DIR "${PROJECT_SOURCE_DIR}/python")
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
message(STATUS "PYPI_PACKAGE_NAME: ${PYPI_PACKAGE_NAME}")


# add the open3d python module first
set( COMPILED_MODULE_PATH_LIST $<TARGET_FILE:${PACKAGE_NAME}> )

set(GENERATED_OUTPUTS "")

# add additional optional compiled modules
if ( BUILD_TENSORFLOW_OPS )
    list( APPEND COMPILED_MODULE_PATH_LIST $<TARGET_FILE:open3d_tf_ops> )
    add_custom_command( OUTPUT "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py"
            COMMAND ${PYTHON_EXECUTABLE} generate_tf_ops_wrapper.py --input "${PYTHON_PACKAGE_SRC_DIR}/open3d/ml/tf/python/ops/ops.py.in" --output "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py" --lib $<TARGET_FILE:open3d_tf_ops>
                        DEPENDS open3d_tf_ops
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        COMMENT "Generating python ops.py" )

    list(APPEND GENERATED_OUTPUTS "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py")
    # find tensorflow to get some info for the _build_config.py
    find_package(Tensorflow)
endif ()

# add additional optional compiled modules
if ( BUILD_PYTORCH_OPS )
    list( APPEND COMPILED_MODULE_PATH_LIST $<TARGET_FILE:open3d_torch_ops> )
    add_custom_command( OUTPUT "${CMAKE_BINARY_DIR}/lib/ml/torch/python/ops.py" "${CMAKE_BINARY_DIR}/lib/ml/torch/python/return_types.py"
            COMMAND ${PYTHON_EXECUTABLE} generate_torch_ops_wrapper.py --input_ops_py_in "${PYTHON_PACKAGE_SRC_DIR}/open3d/ml/torch/python/ops.py.in" --input_return_types_py_in "${PYTHON_PACKAGE_SRC_DIR}/open3d/ml/torch/python/return_types.py.in" --output_dir "${CMAKE_BINARY_DIR}/lib/ml/torch/python/" --lib $<TARGET_FILE:open3d_torch_ops> --tensorflow_ops_dir "${CMAKE_CURRENT_SOURCE_DIR}/../open3d/ml/tensorflow"
                        DEPENDS open3d_torch_ops
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        COMMENT "Generating python ops.py and return_types.py" )

    list(APPEND GENERATED_OUTPUTS "${CMAKE_BINARY_DIR}/lib/ml/torch/python/ops.py" "${CMAKE_BINARY_DIR}/lib/ml/torch/python/return_types.py")
    # get the pytorch version information again here for _build_config.py
    # because it is not safe to call find_package(Pytorch) again.
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
                "import torch; print(torch.__version__, end='')"
        OUTPUT_VARIABLE Pytorch_VERSION)
endif ()

if ( BUNDLE_OPEN3D_ML )

    find_path(
        OPEN3D_ML_ROOT
        NAMES set_open3d_ml_root.sh
        HINTS $ENV{OPEN3D_ML_ROOT}
        DOC "Path to the Open3D-ML repo. This should be set if BUNDLE_OPEN3D_ML is enabled. Alternatively set an env var with the same name to populate this var."
        REQUIRED
        NO_DEFAULT_PATH
    )

    ExternalProject_Add(
        open3d_ml
        PREFIX "${CMAKE_BINARY_DIR}/open3d_ml"
        GIT_REPOSITORY "${OPEN3D_ML_ROOT}"
        BUILD_IN_SOURCE ON
        # do not configure
        CONFIGURE_COMMAND ""
        # do not build
        BUILD_COMMAND ""
        # do not install
        INSTALL_COMMAND ""
        )
    list(APPEND GENERATED_OUTPUTS open3d_ml )
endif ()

configure_file("_build_config.py.in"
               "${CMAKE_BINARY_DIR}/lib/_build_config.py")

add_custom_target(python-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_SRC_DIR=${PYTHON_PACKAGE_SRC_DIR}
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPYTHON_VERSION=${PYTHON_VERSION}
            "-DCOMPILED_MODULE_PATH_LIST=${COMPILED_MODULE_PATH_LIST}"
            "-DPYTHON_EXTRA_LIBRARIES=${PYTHON_EXTRA_LIBRARIES}"
            -DBUILD_JUPYTER_EXTENSION=${BUILD_JUPYTER_EXTENSION}
            -DBUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}
            -DBUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}
            -DBUNDLE_OPEN3D_ML=${BUNDLE_OPEN3D_ML}
            -DOPEN3D_ML_ROOT=${OPEN3D_ML_ROOT}
            -DBUILD_GUI=${BUILD_GUI}
            -DGUI_RESOURCE_DIR=${GUI_RESOURCE_DIR}
            -DPROJECT_EMAIL=${PROJECT_EMAIL}
            -DPROJECT_HOME=${PROJECT_HOME}
            -DPROJECT_DOCS=${PROJECT_DOCS}
            -DPROJECT_CODE=${PROJECT_CODE}
            -DPROJECT_ISSUES=${PROJECT_ISSUES}
            -DPROJECT_VERSION=${OPEN3D_VERSION_FULL}
            -DPROJECT_VERSION_THREE_NUMBER=${PROJECT_VERSION_THREE_NUMBER}
            -DPYPI_PACKAGE_NAME=${PYPI_PACKAGE_NAME}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake
    VERBATIM
    DEPENDS ${GENERATED_OUTPUTS}
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

# FOR DEBUGGING ONLY Use `make install-python-package` to build and install
# python package in the current python environment. This is substantially
# faster than `make install-pip-package`. However this approach does not create
# wheel or egg files and does not take care of dependencies thus not suitable
# for deployment.
# Ref: https://stackoverflow.com/a/33791008/1255535
add_custom_target(install-python-package
    COMMAND ${PYTHON_EXECUTABLE} setup.py install --single-version-externally-managed --root=/
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)

# Use `make conda-package` to create conda package in the build directory
# Note that we don't provide `make install-conda-package` similar to pip. This
# is becuase:
#     1) `make install-pip-package` works in conda environment for local build
#     2) `make conda-package` is mainly for internal use to distribute conda
add_custom_target(conda-package
    COMMAND conda-build conda_meta --channel conda-forge --output-folder conda_package
    COMMAND echo "conda package created at ${PYTHON_PACKAGE_DST_DIR}/conda_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)

# Make pip and conda package in one command. The python-package target clears
# the ${PYTHON_PACKAGE_DST_DIR}, so if we make pip and conda package one-by-one,
# the previous one will be deleted.
add_custom_target(pip-conda-package
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
    COMMAND conda-build conda_meta --channel conda-forge --output-folder conda_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    COMMAND echo "conda package created at ${PYTHON_PACKAGE_DST_DIR}/conda_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)
