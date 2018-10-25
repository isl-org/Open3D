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
)

# 2) The compiled python-C++ module, i.e. open3d.so (or the equivalents)
get_filename_component(PYTHON_COMPILED_MODULE_NAME ${PYTHON_COMPILED_MODULE_PATH} NAME)
file(COPY ${PYTHON_COMPILED_MODULE_PATH}
     DESTINATION ${PYTHON_PACKAGE_DST_DIR}/open3d)

# 3) Configured files and supporting files
configure_file("${PYTHON_PACKAGE_SRC_DIR}/setup.py"
               "${PYTHON_PACKAGE_DST_DIR}/setup.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/open3d/__init__.py"
               "${PYTHON_PACKAGE_DST_DIR}/open3d/__init__.py")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/conda_meta/conda_build_config.yaml"
               "${PYTHON_PACKAGE_DST_DIR}/conda_meta/conda_build_config.yaml")
configure_file("${PYTHON_PACKAGE_SRC_DIR}/conda_meta/meta.yaml"
               "${PYTHON_PACKAGE_DST_DIR}/conda_meta/meta.yaml")
