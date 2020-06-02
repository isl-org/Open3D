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

if (STATIC_WINDOWS_RUNTIME)
    set(WITH_CRT_DLL OFF)
else()
    set(WITH_CRT_DLL ON)
endif()
message(STATUS "libturbojpeg: WITH_CRT_DLL=${WITH_CRT_DLL}")

ExternalProject_Add(
    ext_turbojpeg
    PREFIX turbojpeg
    SOURCE_DIR ${Open3D_3RDPARTY_DIR}/libjpeg-turbo/libjpeg-turbo
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DWITH_CRT_DLL=${WITH_CRT_DLL}
        -DENABLE_STATIC=ON
        -DENABLE_SHARED=OFF
        -DWITH_SIMD=${WITH_SIMD}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-install
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

# If MSVC, the OUTPUT_NAME was set to turbojpeg-static
if(MSVC)
    set(lib_name "turbojpeg-static")
else()
    set(lib_name "turbojpeg")
endif()

# For linking with Open3D's after installation
set(JPEG_TURBO_LIBRARIES ${lib_name})
