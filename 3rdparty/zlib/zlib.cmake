include(ExternalProject)

if(MSVC)
    set(lib_name zlibstatic)
else()
    set(lib_name z)
endif()

find_package(Git QUIET REQUIRED)

ExternalProject_Add(
    ext_zlib
    PREFIX zlib
    URL https://github.com/madler/zlib/archive/refs/tags/v1.2.13.tar.gz
    URL_HASH SHA256=1525952a0a567581792613a9723333d7f8cc20b87a81f920fb8bc7e3f2251428
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/zlib"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${GIT_EXECUTABLE} init
    COMMAND       ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
                  ${CMAKE_CURRENT_LIST_DIR}/0001-patch-zlib-to-enable-unzip.patch
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        # zlib needs visiible symbols for examples. Disabling example building causes
        # assember error in GPU CI. zlib symbols are hidden during linking.
        ${ExternalProject_CMAKE_ARGS}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_zlib INSTALL_DIR)
set(ZLIB_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(ZLIB_LIB_DIR ${INSTALL_DIR}/lib)
if(MSVC)
    set(ZLIB_LIBRARIES ${lib_name}$<$<CONFIG:Debug>:d>)
else()
    set(ZLIB_LIBRARIES ${lib_name})
endif()
