include(ExternalProject)

if(ZLIB_LIB_DIR AND ZLIB_LIBRARIES)
    set(spz_zlib_library
        ${ZLIB_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    set(spz_zlib_library ${ZLIB_LIBRARY})
endif()

ExternalProject_Add(
    ext_spz
    PREFIX spz
    URL https://codeload.github.com/nianticlabs/spz/tar.gz/refs/tags/v3.0.0
    URL_HASH SHA256=266c144fddb87f428495bbe1e44748fc3f1e5bfaa9e354a5d0d0284c4f8946cc
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/spz"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR=${Open3D_INSTALL_LIB_DIR}
        -DSPZ_BUILD_PYTHON_BINDINGS=OFF
        -DSPZ_BUILD_TOOLS=OFF
        -DSPZ_BUILD_EXTENSIONS=OFF
        -DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIRS}
        -DZLIB_LIBRARY:FILEPATH=${spz_zlib_library}
        -DCMAKE_PREFIX_PATH:PATH=${ZSTD_INSTALL_DIR}
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}spz${CMAKE_STATIC_LIBRARY_SUFFIX}
    DEPENDS
        ext_zlib
        ext_zstd
)

ExternalProject_Get_Property(ext_spz INSTALL_DIR)
set(SPZ_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(SPZ_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(SPZ_LIBRARIES spz)
