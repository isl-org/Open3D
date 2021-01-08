include(ExternalProject)

ExternalProject_Add(
    ext_libpng
    PREFIX libpng
    GIT_REPOSITORY https://github.com/glennrp/libpng.git
    GIT_TAG v1.6.37
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DPNG_SHARED=OFF
        -DPNG_EXECUTABLES=OFF
        -DPNG_TESTS=OFF
        -DPNG_BUILD_ZLIB=ON # Prevent libpng from calling find_pacakge(zlib).
        -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIRS}
    BUILD_ALWAYS ON
)

ExternalProject_Get_Property(ext_libpng INSTALL_DIR)
set(LIBPNG_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(LIBPNG_LIB_DIR ${INSTALL_DIR}/lib)
if(MSVC)
    set(LIBPNG_LIBRARIES libpng16_static$<$<CONFIG:Debug>:d>)
else()
    set(LIBPNG_LIBRARIES png16)
endif()
