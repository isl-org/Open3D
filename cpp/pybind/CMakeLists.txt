if (NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python 3 not found in top level file")
endif()

# We need to get python version to configure some meta files
set(PYTHON_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")


pybind11_add_module(pybind)

add_subdirectory(camera)
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(ml)
add_subdirectory(pipelines)
add_subdirectory(t)
add_subdirectory(t/geometry)
add_subdirectory(t/io)
add_subdirectory(t/pipelines)
add_subdirectory(utility)
add_subdirectory(visualization)

target_sources(pybind PRIVATE
    docstring.cpp
    open3d_pybind.cpp
    pybind_utils.cpp
)

# Include with `#include "pybind/geometry/xxx.h`
target_include_directories(pybind PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Suppress Pybind11 warnings
target_include_directories(pybind SYSTEM PRIVATE
    ${PYBIND11_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

open3d_show_and_abort_on_warning(pybind)
open3d_set_global_properties(pybind)

if (WIN32)
    target_link_options(pybind PUBLIC "/force:multiple")
endif()

target_link_libraries(pybind PRIVATE Open3D::Open3D)

# At `make`: open3d.so (or the equivalents) will be created at
# PYTHON_COMPILED_MODULE_DIR. The default location is
# `build/lib/${CMAKE_BUILD_TYPE}/Python/{cpu|cuda}`
set(PYTHON_COMPILED_MODULE_DIR
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Python/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
set_target_properties(pybind PROPERTIES
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
set(COMPILED_MODULE_PATH_LIST $<TARGET_FILE:pybind>)

set(GENERATED_OUTPUTS "")

# add additional optional compiled modules
if (BUILD_TENSORFLOW_OPS)
    list( APPEND COMPILED_MODULE_PATH_LIST $<TARGET_FILE:open3d_tf_ops> )
    add_custom_command( OUTPUT "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py"
            COMMAND ${Python3_EXECUTABLE} generate_tf_ops_wrapper.py --input "${PYTHON_PACKAGE_SRC_DIR}/open3d/ml/tf/python/ops/ops.py.in" --output "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py" --lib $<TARGET_FILE:open3d_tf_ops>
                        DEPENDS open3d_tf_ops
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        COMMENT "Generating python ops.py" )

    list(APPEND GENERATED_OUTPUTS "${CMAKE_BINARY_DIR}/lib/ml/tf/python/ops/ops.py")
    # find tensorflow to get some info for the _build_config.py
    find_package(Tensorflow)
endif()

# add additional optional compiled modules
if (BUILD_PYTORCH_OPS)
    list( APPEND COMPILED_MODULE_PATH_LIST $<TARGET_FILE:open3d_torch_ops> )
    add_custom_command( OUTPUT "${CMAKE_BINARY_DIR}/lib/ml/torch/python/ops.py" "${CMAKE_BINARY_DIR}/lib/ml/torch/python/return_types.py"
            COMMAND ${Python3_EXECUTABLE} generate_torch_ops_wrapper.py --input_ops_py_in "${PYTHON_PACKAGE_SRC_DIR}/open3d/ml/torch/python/ops.py.in" --input_return_types_py_in "${PYTHON_PACKAGE_SRC_DIR}/open3d/ml/torch/python/return_types.py.in" --output_dir "${CMAKE_BINARY_DIR}/lib/ml/torch/python/" --lib $<TARGET_FILE:open3d_torch_ops> --tensorflow_ops_dir "${CMAKE_CURRENT_SOURCE_DIR}/../open3d/ml/tensorflow"
                        DEPENDS open3d_torch_ops
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        COMMENT "Generating python ops.py and return_types.py" )

    list(APPEND GENERATED_OUTPUTS "${CMAKE_BINARY_DIR}/lib/ml/torch/python/ops.py" "${CMAKE_BINARY_DIR}/lib/ml/torch/python/return_types.py")
    # get the pytorch version information again here for _build_config.py
    # because it is not safe to call find_package(Pytorch) again.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "-c"
                "import torch; print(torch.__version__, end='')"
        OUTPUT_VARIABLE Pytorch_VERSION)
endif()

if (BUNDLE_OPEN3D_ML)
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
    list(APPEND GENERATED_OUTPUTS open3d_ml)
endif()

configure_file("_build_config.py.in"
               "${CMAKE_BINARY_DIR}/lib/_build_config.py.in")
file(GENERATE
    OUTPUT "${PYTHON_COMPILED_MODULE_DIR}/_build_config.py"
    INPUT "${CMAKE_BINARY_DIR}/lib/_build_config.py.in"
)

add_custom_target(python-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_SRC_DIR=${PYTHON_PACKAGE_SRC_DIR}
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPYTHON_COMPILED_MODULE_DIR=${PYTHON_COMPILED_MODULE_DIR}
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
            -DPROJECT_HOMEPAGE_URL=${PROJECT_HOMEPAGE_URL}
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
    COMMAND ${Python3_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)

# Use `make install-pip-package` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
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
    COMMAND ${Python3_EXECUTABLE} setup.py install --single-version-externally-managed --root=/
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
    COMMAND ${Python3_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
    COMMAND conda-build conda_meta --channel conda-forge --output-folder conda_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    COMMAND echo "conda package created at ${PYTHON_PACKAGE_DST_DIR}/conda_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)
