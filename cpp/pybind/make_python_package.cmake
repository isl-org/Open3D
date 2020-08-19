# Clean up directory
file(REMOVE_RECURSE ${PYTHON_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/open3d)

# Create python pacakge. It contains:
# 1) Pure-python code and misc files, copied from ${PYTHON_PACKAGE_SRC_DIR}
# 2) The compiled python-C++ module, i.e. open3d.so (or the equivalents)
#    Optionally other modules e.g. open3d_tf_ops.so may be included.
# 3) Configured files and supporting files

# 1) Pure-python code and misc files, copied from ${PYTHON_PACKAGE_SRC_DIR}
file(COPY ${PYTHON_PACKAGE_SRC_DIR}/
     DESTINATION ${PYTHON_PACKAGE_DST_DIR}
)

# 2) The compiled python-C++ module, i.e. open3d.so (or the equivalents)
#    Optionally other modules e.g. open3d_tf_ops.so may be included.
list(POP_FRONT COMPILED_MODULE_PATH_LIST PYTHON_COMPILED_MODULE_PATH )
get_filename_component(PYTHON_COMPILED_MODULE_NAME ${PYTHON_COMPILED_MODULE_PATH} NAME)

# For CUDA+CPU package, copy multiple pybind*.so compiled modules in separate
# architecture folders

get_filename_component(PYTHON_COMPILED_MODULE_DIR
    ${PYTHON_COMPILED_MODULE_PATH} DIRECTORY)
message(STATUS "Copying ${PYTHON_COMPILED_MODULE_DIR} to ${PYTHON_PACKAGE_DST_DIR}/open3d")
file(COPY ${PYTHON_COMPILED_MODULE_DIR}
    DESTINATION ${PYTHON_PACKAGE_DST_DIR}/open3d)
if (${PYTHON_COMPILED_MODULE_DIR} MATCHES ".*/cuda")
    string(REPLACE "/cuda" "/cpu" PYTHON_COMPILED_MODULE_DIR2
        ${PYTHON_COMPILED_MODULE_DIR})
else()
    string(REPLACE "/cpu" "/cuda" PYTHON_COMPILED_MODULE_DIR2
        ${PYTHON_COMPILED_MODULE_DIR})
endif()
if(EXISTS ${PYTHON_COMPILED_MODULE_DIR2})
    message(STATUS "Copying ${PYTHON_COMPILED_MODULE_DIR2} to ${PYTHON_PACKAGE_DST_DIR}/open3d")
    file(COPY ${PYTHON_COMPILED_MODULE_DIR2}
        DESTINATION ${PYTHON_PACKAGE_DST_DIR}/open3d)
endif()

file(COPY ${COMPILED_MODULE_PATH_LIST}
     DESTINATION ${PYTHON_PACKAGE_DST_DIR}/open3d)

# 3) Configured files and supporting files
configure_file("${PYTHON_PACKAGE_SRC_DIR}/setup.py"
               "${PYTHON_PACKAGE_DST_DIR}/setup.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/open3d/__init__.py"
               "${PYTHON_PACKAGE_DST_DIR}/open3d/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/open3d/visualization/__init__.py"
               "${PYTHON_PACKAGE_DST_DIR}/open3d/visualization/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/open3d/visualization/gui/__init__.py"
               "${PYTHON_PACKAGE_DST_DIR}/open3d/visualization/gui/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/open3d/visualization/rendering/__init__.py"
               "${PYTHON_PACKAGE_DST_DIR}/open3d/visualization/rendering/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/open3d/j_visualizer.py"
               "${PYTHON_PACKAGE_DST_DIR}/open3d/j_visualizer.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/conda_meta/conda_build_config.yaml"
               "${PYTHON_PACKAGE_DST_DIR}/conda_meta/conda_build_config.yaml")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/conda_meta/meta.yaml"
               "${PYTHON_PACKAGE_DST_DIR}/conda_meta/meta.yaml")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/js/j_visualizer.js"
               "${PYTHON_PACKAGE_DST_DIR}/js/j_visualizer.js")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/js/package.json"
               "${PYTHON_PACKAGE_DST_DIR}/js/package.json")
file(COPY "${PYTHON_PACKAGE_DST_DIR}/../_build_config.py"
     DESTINATION "${PYTHON_PACKAGE_DST_DIR}/open3d/" )

if (BUILD_TENSORFLOW_OPS OR BUILD_PYTORCH_OPS)
    # copy generated files
    file(COPY "${PYTHON_PACKAGE_DST_DIR}/../ml"
         DESTINATION "${PYTHON_PACKAGE_DST_DIR}/open3d/" )
endif()

# Build Jupyter plugin with webpack. This step distills and merges all js
# dependencies and include all static assets. The generated output is in
# ${PYTHON_PACKAGE_DST_DIR}/open3d/static.
if (BUILD_JUPYTER_EXTENSION)
    file(REMOVE_RECURSE ${PYTHON_PACKAGE_DST_DIR}/open3d/static)
    message(STATUS "Jupyter support is enabled. Building Jupyter plugin ...")
    if (WIN32)
        find_program(NPM "npm")
        execute_process(
            COMMAND cmd /c "${NPM}" install
            RESULT_VARIABLE res_var
            WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/js
        )
    else()
        execute_process(
            COMMAND npm install
            RESULT_VARIABLE res_var
            WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}/js
        )
    endif()
    if (NOT "${res_var}" STREQUAL "0")
        message(FATAL_ERROR "`npm install` failed with: '${res_var}'")
    endif()

    # We cache ${PYTHON_PACKAGE_DST_DIR}/js/node_modules in
    #          ${PYTHON_PACKAGE_SRC_DIR}/js/node_modules
    # to speed up webpack build speed during development.
    # During build, the following steps will happen:
    # 1) The entire ${PYTHON_PACKAGE_DST_DIR} in the build directory is cleared.
    # 2) ${PYTHON_PACKAGE_SRC_DIR}/js/node_modules is copied to
    #    ${PYTHON_PACKAGE_DST_DIR}/js/node_modules, regadless whether
    #    ${PYTHON_PACKAGE_SRC_DIR}/js/node_modules is empty or not.
    # 3) `npm install` is run in ${PYTHON_PACKAGE_DST_DIR}/js, so
    #    ${PYTHON_PACKAGE_DST_DIR}/js/node_modules must be filled after
    #    `npm install`.
    # 4) ${PYTHON_PACKAGE_DST_DIR}/js/node_modules is then copied back to
    #    ${PYTHON_PACKAGE_SRC_DIR}/js/node_modules for caching.
    file(REMOVE_RECURSE ${PYTHON_PACKAGE_SRC_DIR}/js/node_modules)
    file(COPY ${PYTHON_PACKAGE_DST_DIR}/js/node_modules
        DESTINATION ${PYTHON_PACKAGE_SRC_DIR}/js)
endif()

if (BUILD_GUI)
    file(MAKE_DIRECTORY "${PYTHON_PACKAGE_DST_DIR}/open3d/resources/")
    file(COPY ${GUI_RESOURCE_DIR}
         DESTINATION "${PYTHON_PACKAGE_DST_DIR}/open3d/")
endif()
