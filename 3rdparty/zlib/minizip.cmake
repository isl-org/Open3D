include(ExternalProject)

find_package(Git QUIET REQUIRED)

ExternalProject_Get_Property(ext_zlib INSTALL_DIR)
set(ZLIB_INSTALL_DIR ${INSTALL_DIR})
unset(INSTALL_DIR)

ExternalProject_Add(ext_minizip
    PREFIX minizip-ng
    URL https://github.com/zlib-ng/minizip-ng/archive/refs/tags/4.0.7.tar.gz
    URL_HASH SHA256=a87f1f734f97095fe1ef0018217c149d53d0f78438bcb77af38adc21dff2dfbc
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/minizip-ng"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DMZ_COMPAT:BOOL=ON
        -DMZ_ZLIB:BOOL=ON
        -DMZ_BZIP2:BOOL=OFF
        -DMZ_LZMA:BOOL=OFF
        -DMZ_ZSTD:BOOL=OFF
        -DMZ_LIBCOMP:BOOL=OFF
        -DMZ_FETCH_LIBS:BOOL=OFF
        -DMZ_FORCE_FETCH_LIBS:BOOL=OFF
        -DMZ_PKCRYPT:BOOL=OFF
        -DMZ_WZAES:BOOL=OFF
        -DMZ_OPENSSL:BOOL=OFF
        -DMZ_LIBBSD:BOOL=OFF
        -DMZ_ICONV:BOOL=OFF
        -DZLIB_ROOT:PATH=${ZLIB_INSTALL_DIR}
        ${ExternalProject_CMAKE_ARGS}
)

set(lib_name minizip)
ExternalProject_Get_Property(ext_minizip INSTALL_DIR)
set(MINIZIP_INCLUDE_DIRS ${INSTALL_DIR}/include/minizip/) # "/" is critical.
set(MINIZIP_LIB_DIR ${INSTALL_DIR}/lib)
set(MINIZIP_LIBRARIES ${lib_name})
