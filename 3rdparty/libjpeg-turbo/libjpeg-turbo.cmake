include(ExternalProject)

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

# Set compiler flags
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
endif()

# Set WITH_SIMD
include(CheckLanguage)
check_language(ASM_NASM)
if (CMAKE_ASM_NASM_COMPILER)
    if (APPLE)
        # macOS might have /usr/bin/nasm but it cannot be used
        # https://stackoverflow.com/q/53974320
        # To fix this, run `brew install nasm`
        execute_process(COMMAND nasm --version RESULT_VARIABLE return_code)
        if("${return_code}" STREQUAL "0")
            enable_language(ASM_NASM)
            option(WITH_SIMD "" ON)
        else()
            message(STATUS "nasm found but can't be used, run `brew install nasm`")
            option(WITH_SIMD "" OFF)
        endif()
    else()
        enable_language(ASM_NASM)
        option(WITH_SIMD "" ON)
    endif()
else()
    option(WITH_SIMD "" OFF)
endif()
if (WITH_SIMD)
    message(STATUS "NASM assembler enabled")
else()
    message(STATUS "NASM assembler not found - libjpeg-turbo performance may suffer")
endif()

ExternalProject_Add(
    ext_turbojpeg
    PREFIX turbojpeg
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo/libjpeg-turbo
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${DCMAKE_C_FLAGS}
        -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
        -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
        -DENABLE_STATIC=ON
        -DENABLE_SHARED=OFF
        -DWITH_SIMD=${WITH_SIMD}
        -DCMAKE_INSTALL_PREFIX=${3RDPARTY_INSTALL_PREFIX}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

add_library(turbojpeg INTERFACE)
add_dependencies(turbojpeg ext_turbojpeg)

# If MSVC, the OUTPUT_NAME was set to turbojpeg-static
if(MSVC)
    set(lib_name "turbojpeg-static")
else()
    set(lib_name "turbojpeg")
endif()

# For linking with Open3D's after installation
set(JPEG_TURBO_LIBRARIES ${lib_name})

set(turbojpeg_LIB_FILES
    ${3RDPARTY_INSTALL_PREFIX}/${LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
)

target_include_directories(turbojpeg SYSTEM INTERFACE
    ${3RDPARTY_INSTALL_PREFIX}/include
)
target_link_libraries(turbojpeg INTERFACE
    ${turbojpeg_LIB_FILES}
)

if (NOT BUILD_SHARED_LIBS)
    install(FILES ${turbojpeg_LIB_FILES}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

add_dependencies(build_all_3rd_party_libs turbojpeg)


