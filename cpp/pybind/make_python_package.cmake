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
# Folder structure is base_dir/{cpu|cuda}/{pybind*.so|open3d_{torch|tf}_ops.so},
# so copy base_dir directly to ${PYTHON_PACKAGE_DST_DIR}/open3d
foreach(COMPILED_MODULE_PATH ${COMPILED_MODULE_PATH_LIST})
    get_filename_component(COMPILED_MODULE_NAME ${COMPILED_MODULE_PATH} NAME)
    get_filename_component(COMPILED_MODULE_ARCH_DIR ${COMPILED_MODULE_PATH} DIRECTORY)
    get_filename_component(COMPILED_MODULE_BASE_DIR ${COMPILED_MODULE_ARCH_DIR} DIRECTORY)
    foreach(ARCH cpu cuda)
        if(IS_DIRECTORY "${COMPILED_MODULE_BASE_DIR}/${ARCH}")
            file(INSTALL "${COMPILED_MODULE_BASE_DIR}/${ARCH}/" DESTINATION
                "${PYTHON_PACKAGE_DST_DIR}/open3d/${ARCH}"
                FILES_MATCHING PATTERN "${COMPILED_MODULE_NAME}")
        endif()
    endforeach()
endforeach()
# Include additional libraries that may be absent from the user system
# eg: libc++.so and libc++abi.so (needed by filament)
# The linker recognizes only library.so.MAJOR, so remove .MINOR from the filname
foreach(PYTHON_EXTRA_LIB ${PYTHON_EXTRA_LIBRARIES})
    get_filename_component(PYTHON_EXTRA_LIB_REAL ${PYTHON_EXTRA_LIB} REALPATH)
    get_filename_component(SO_VER_NAME ${PYTHON_EXTRA_LIB_REAL} NAME)
    string(REGEX REPLACE "\\.so\\.1\\..*" ".so.1" SO_1_NAME ${SO_VER_NAME})
    configure_file(${PYTHON_EXTRA_LIB_REAL} ${PYTHON_PACKAGE_DST_DIR}/open3d/${SO_1_NAME} COPYONLY)
endforeach()

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

if (BUNDLE_OPEN3D_ML)
    file(COPY "${PYTHON_PACKAGE_DST_DIR}/../../open3d_ml/src/open3d_ml/ml3d"
         DESTINATION "${PYTHON_PACKAGE_DST_DIR}/open3d/" )
    file(RENAME "${PYTHON_PACKAGE_DST_DIR}/open3d/ml3d" "${PYTHON_PACKAGE_DST_DIR}/open3d/_ml3d")
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
