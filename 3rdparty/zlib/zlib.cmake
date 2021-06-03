include(ExternalProject)

if(MSVC)
    set(lib_name zlibstatic)
else()
    set(lib_name z)
endif()

ExternalProject_Add(
    ext_zlib
    PREFIX zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.2.11
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        ${ExternalProject_CMAKE_ARGS}  # Symbols not hidden due to examples built
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
