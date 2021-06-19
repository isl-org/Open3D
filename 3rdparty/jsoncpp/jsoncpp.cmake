include(ExternalProject)

find_package(Git QUIET REQUIRED)

ExternalProject_Add(
    ext_jsoncpp
    PREFIX jsoncpp
    URL https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/1.9.4.tar.gz
    URL_HASH SHA256=e34a628a8142643b976c7233ef381457efad79468c67cb1ae0b83a33d7493999
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/jsoncpp"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${GIT_EXECUTABLE} init
    COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
        ${CMAKE_CURRENT_LIST_DIR}/0001-optional-CXX11-ABI-and-MSVC-runtime.patch
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_STATIC_LIBS=ON
        -DBUILD_OBJECT_LIBS=OFF
        -DJSONCPP_WITH_TESTS=OFF
        -DJSONCPP_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}
        -DJSONCPP_STATIC_WINDOWS_RUNTIME=${STATIC_WINDOWS_RUNTIME}
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}jsoncpp${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_jsoncpp INSTALL_DIR)
set(JSONCPP_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(JSONCPP_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(JSONCPP_LIBRARIES jsoncpp)
