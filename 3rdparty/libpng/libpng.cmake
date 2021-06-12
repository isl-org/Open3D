include(ExternalProject)

if(MSVC)
    set(lib_name libpng16_static)
else()
    set(lib_name png16)
endif()

ExternalProject_Add(
    ext_libpng
    PREFIX libpng
    GIT_REPOSITORY https://github.com/glennrp/libpng.git
    GIT_TAG v1.6.37
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DPNG_SHARED=OFF
        -DPNG_EXECUTABLES=OFF
        -DPNG_TESTS=OFF
        -DPNG_BUILD_ZLIB=ON # Prevent libpng from calling find_pacakge(zlib).
        -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIRS}
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_libpng INSTALL_DIR)
set(LIBPNG_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(LIBPNG_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(LIBPNG_LIBRARIES ${lib_name}$<$<CONFIG:Debug>:d>)
