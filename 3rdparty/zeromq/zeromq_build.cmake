include(ExternalProject)
include(FetchContent)

# ExternalProject seems to be the best solution for including zeromq.
# The projects defines options which clash with and pollute our CMake cache.

# Define the compile flags for Windows
if(WIN32)
    set(WIN_CMAKE_ARGS "-DCMAKE_CXX_FLAGS_DEBUG=$<IF:$<BOOL:${STATIC_WINDOWS_RUNTIME}>,/MTd,/MDd> ${CMAKE_CXX_FLAGS_DEBUG_INIT}"
                       "-DCMAKE_CXX_FLAGS_RELEASE=$<IF:$<BOOL:${STATIC_WINDOWS_RUNTIME}>,/MT,/MD> ${CMAKE_CXX_FLAGS_RELEASE_INIT}"
                       "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=$<IF:$<BOOL:${STATIC_WINDOWS_RUNTIME}>,/MT,/MD> ${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT}"
                       "-DCMAKE_CXX_FLAGS_MINSIZEREL=$<IF:$<BOOL:${STATIC_WINDOWS_RUNTIME}>,/MT,/MD> ${CMAKE_CXX_FLAGS_MINSIZEREL_INIT}"
                       "-DCMAKE_C_FLAGS_DEBUG=$<IF:$<BOOL:${STATIC_WINDOWS_RUNTIME}>,/MTd,/MDd> ${CMAKE_C_FLAGS_DEBUG_INIT}"
                       "-DCMAKE_C_FLAGS_RELEASE=$<IF:$<BOOL:${STATIC_WINDOWS_RUNTIME}>,/MT,/MD> ${CMAKE_C_FLAGS_RELEASE_INIT}"
                       "-DCMAKE_C_FLAGS_RELWITHDEBINFO=$<IF:$<BOOL:${STATIC_WINDOWS_RUNTIME}>,/MT,/MD> ${CMAKE_C_FLAGS_RELWITHDEBINFO_INIT}"
                       "-DCMAKE_C_FLAGS_MINSIZEREL=$<IF:$<BOOL:${STATIC_WINDOWS_RUNTIME}>,/MT,/MD> ${CMAKE_C_FLAGS_MINSIZEREL_INIT}"
                       )
    set(lib_name libzmq)
    if(CMAKE_VS_PLATFORM_TOOLSET)
        string(APPEND lib_name -${CMAKE_VS_PLATFORM_TOOLSET})
    endif()
    string(APPEND lib_name -mt-s)
    set(lib_suffix -4_3_3)
else()
    set(WIN_CMAKE_ARGS "")
    set(lib_name zmq)
endif()


ExternalProject_Add(
    ext_zeromq
    PREFIX "${CMAKE_BINARY_DIR}/zeromq"
    URL "https://github.com/zeromq/libzmq/releases/download/v4.3.3/zeromq-4.3.3.tar.gz"
    URL_HASH MD5=78acc277d95e10812d71b2b3c3c3c9a9
    # do not update
    UPDATE_COMMAND ""
    CMAKE_ARGS
        # Does not seem to work. We have to directly set the flags on Windows.
        #-DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW
        #-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
        -DBUILD_STATIC=ON
        -DBUILD_SHARED=OFF
        -DBUILD_TESTS=OFF
        -DENABLE_CPACK=OFF
        -DENABLE_CURVE=OFF
        -DZMQ_BUILD_TESTS=OFF
        -DWITH_DOCS=OFF
        -DWITH_PERF_TOOL=OFF
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        ${WIN_CMAKE_ARGS}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${lib_suffix}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}gd${lib_suffix}${CMAKE_STATIC_LIBRARY_SUFFIX}
)

# cppzmq is header only. we just need to download
FetchContent_Declare(
    ext_cppzmq
    URL "https://github.com/zeromq/cppzmq/archive/v4.7.1.tar.gz"
    URL_HASH MD5=e85cf23b5aed263c2c5c89657737d107
)
FetchContent_GetProperties(ext_cppzmq)
if(NOT ext_cppzmq_POPULATED)
    FetchContent_Populate(ext_cppzmq)
    # do not add subdirectory here
endif()

if(WIN32)
    # On windows the lib name is more complicated
    set(ZEROMQ_LIBRARIES ${lib_name}$<$<CONFIG:Debug>:gd>${lib_suffix})

    # On windows we need to link some additional libs. We will use them
    # directly as targets in find_dependencies.cmake.
    # The following code is taken from the zeromq CMakeLists.txt and collects
    # the additional libs in ZEROMQ_ADDITIONAL_LIBS.
    include(CheckCXXSymbolExists)
    set(CMAKE_REQUIRED_LIBRARIES "ws2_32.lib")
    check_cxx_symbol_exists(WSAStartup "winsock2.h" HAVE_WS2_32)

    set(CMAKE_REQUIRED_LIBRARIES "rpcrt4.lib")
    check_cxx_symbol_exists(UuidCreateSequential "rpc.h" HAVE_RPCRT4)

    set(CMAKE_REQUIRED_LIBRARIES "iphlpapi.lib")
    check_cxx_symbol_exists(GetAdaptersAddresses "winsock2.h;iphlpapi.h" HAVE_IPHLPAPI)
    set(CMAKE_REQUIRED_LIBRARIES "")

    if(HAVE_WS2_32)
        list(APPEND ZEROMQ_ADDITIONAL_LIBS ws2_32)
    endif()
    if(HAVE_RPCRT4)
        list(APPEND ZEROMQ_ADDITIONAL_LIBS rpcrt4)
    endif()
    if(HAVE_IPHLPAPI)
        list(APPEND ZEROMQ_ADDITIONAL_LIBS iphlpapi)
    endif()

else()
    set(ZEROMQ_LIBRARIES ${lib_name}${lib_suffix})
endif()
ExternalProject_Get_Property( ext_zeromq INSTALL_DIR )
set(ZEROMQ_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(ZEROMQ_INCLUDE_DIRS "${INSTALL_DIR}/include/;${ext_cppzmq_SOURCE_DIR}/")
