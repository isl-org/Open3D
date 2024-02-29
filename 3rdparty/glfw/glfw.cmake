include(ExternalProject)

set(GLFW_LIB_NAME glfw3)

ExternalProject_Add(
    ext_glfw
    PREFIX glfw
    URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
    URL_HASH SHA256=c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/glfw"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DGLFW_BUILD_EXAMPLES=OFF
        -DGLFW_BUILD_TESTS=OFF
        -DGLFW_BUILD_DOCS=OFF
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${GLFW_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_glfw INSTALL_DIR)
set(GLFW_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(GLFW_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(GLFW_LIBRARIES ${GLFW_LIB_NAME})
