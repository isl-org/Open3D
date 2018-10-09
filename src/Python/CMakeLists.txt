if (NOT PYTHON_EXECUTABLE)
    # find_package(PythonInterp REQUIRED) returns the system python
    # find_program will returns the python executable in current PATH, which
    # works with virtualenv
    # Note: make sure to run `cmake ..` within your virtual environment!
    find_program(PYTHON_IN_PATH "python")
    set(PYTHON_EXECUTABLE ${PYTHON_IN_PATH})
endif()
message("Using Python executable ${PYTHON_EXECUTABLE}")

set(PACKAGE_NAME open3d)

file(GLOB_RECURSE PY_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE PY_ALL_SOURCE_FILES "*.cpp")

ADD_SOURCE_GROUP(Core)
ADD_SOURCE_GROUP(IO)
ADD_SOURCE_GROUP(Tutorial)
ADD_SOURCE_GROUP(Visualization)
pybind11_add_module(${PACKAGE_NAME}
    ${PY_ALL_HEADER_FILES} ${PY_ALL_SOURCE_FILES}
    ${PY_CORE_SOURCE_FILES} ${PY_CORE_HEADER_FILES}
    ${PY_IO_SOURCE_FILES} ${PY_IO_HEADER_FILES}
    ${PY_VISUALIZATION_SOURCE_FILES} ${PY_VISUALIZATION_HEADER_FILES}
)

target_link_libraries(${PACKAGE_NAME} PRIVATE ${CMAKE_PROJECT_NAME})

# At `make`: open3d.so (or the equivalents) will be created at
# PYTHON_COMPILED_MODULE_DIR. The default locaiton is `build/lib/Python`
set(PYTHON_COMPILED_MODULE_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Python")
set_target_properties(${PACKAGE_NAME} PROPERTIES
                      FOLDER "Python"
                      LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
                      ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")

# Use `make python-package` to create the python package in the build directory
# The python package will be created at PYTHON_PACKAGE_DIR. It contains:
# 1) Pure-python code and misc files, copied from src/Python/Package
# 2) The compiled python-C++ module, i.e. open3d.so (or the equivalents)
# 3) Configured files and supporting files
# Note: `make python-package` clears PYTHON_COMPILED_MODULE_DIR first every time
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
add_custom_target(python-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/Package
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPYTHON_COMPILED_MODULE_PATH=$<TARGET_FILE:${PACKAGE_NAME}>
            -DPROJECT_EMAIL=${PROJECT_EMAIL}
            -DPROJECT_HOME=${PROJECT_HOME}
            -DPROJECT_DOCS=${PROJECT_DOCS}
            -DPROJECT_CODE=${PROJECT_CODE}
            -DPROJECT_ISSUES=${PROJECT_ISSUES}
            -DPROJECT_VERSION=${PROJECT_VERSION}
            -DPYPI_PACKAGE_NAME=open3d  # `open3d` is used in building from source
            -P ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake
)

# Use `make pip-wheel` to create the pip package in the build directory
add_custom_target(pip-wheel
    COMMAND ${PYTHON_EXECUTABLE} "setup.py" "bdist_wheel"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)

# Use `make install-pip-wheel` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-wheel
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/install_pip_wheel.cmake
    DEPENDS pip-wheel
)

################################################################################
# Warning: Internal use only, consider droping this in the future
# Use `make all-pip-wheels` to create the pip package in the build directory
# This creates: open3d-python, py3d, open3d-original, open3d-official, open-3d
#               pip wheels
add_custom_target(all-pip-wheels
    COMMAND ${CMAKE_COMMAND}
        -DPYTHON_PACKAGE_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/Package
        -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
        -DPYTHON_COMPILED_MODULE_PATH=$<TARGET_FILE:${PACKAGE_NAME}>
        -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
        -DPROJECT_EMAIL=${PROJECT_EMAIL}
        -DPROJECT_HOME=${PROJECT_HOME}
        -DPROJECT_DOCS=${PROJECT_DOCS}
        -DPROJECT_CODE=${PROJECT_CODE}
        -DPROJECT_ISSUES=${PROJECT_ISSUES}
        -DPROJECT_VERSION=${PROJECT_VERSION}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/make_all_pip_wheels.cmake
)
