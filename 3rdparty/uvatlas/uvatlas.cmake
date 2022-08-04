include(ExternalProject)

# uvatlas needs some headers from directx 
ExternalProject_Add(
    ext_directxheaders
    PREFIX uvatlas
    GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers.git
    GIT_TAG v1.606.3
    GIT_SHALLOW TRUE
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/uvatlas"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_GMOCK=OFF
        -DDXHEADERS_BUILD_GOOGLE_TEST=OFF
        -DDXHEADERS_BUILD_TEST=OFF
        -DINSTALL_GTEST=OFF
)

# uvatlas needs DirectXMath
ExternalProject_Add(
    ext_directxmath
    PREFIX uvatlas
    GIT_REPOSITORY https://github.com/microsoft/DirectXMath.git
    GIT_TAG may2022
    GIT_SHALLOW TRUE
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/uvatlas"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)

# uvatlas
ExternalProject_Add(
    ext_uvatlas
    PREFIX uvatlas
    URL https://github.com/microsoft/UVAtlas/archive/refs/tags/may2022.tar.gz
    URL_HASH SHA256=591516913a0f3c381f1fd01647cb1b8d1eeade575d1c726ae8f5dd9f83b81754
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/uvatlas"
    # do not update
    UPDATE_COMMAND ""
    # copy a dummy sal.h to the include dir. (Visual Studio source-code annotation language)
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/uvatlas/sal.h <INSTALL_DIR>/include/DirectXMath/
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_TOOLS=OFF
        -Ddirectx-headers_DIR=<INSTALL_DIR>
        -Ddirectxmath_DIR=<INSTALL_DIR>
    DEPENDS ext_directxheaders ext_directxmath
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}uvatlas${CMAKE_STATIC_LIBRARY_SUFFIX}
)


ExternalProject_Get_Property(ext_uvatlas INSTALL_DIR)

set(UVATLAS_INCLUDE_DIRS "${INSTALL_DIR}/include/DirectXMath/" "${INSTALL_DIR}/include/" ) # dont forget trailing '/'
if(NOT WIN32)
    list(APPEND UVATLAS_INCLUDE_DIRS "${INSTALL_DIR}/include/wsl/stubs/")
endif()
set(UVATLAS_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(UVATLAS_LIBRARIES UVAtlas)