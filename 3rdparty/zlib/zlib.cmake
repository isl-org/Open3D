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
        -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
        -DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
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
