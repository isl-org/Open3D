if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
endif()

set(ENABLE_STATIC ON CACHE BOOL "Force libjpeg to enable static lib" FORCE)

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
    message(WARNING "NASM assembler not found - libjpeg-turbo performance may suffer")
endif()

# This generates turbojpeg-static target
add_subdirectory(libjpeg-turbo EXCLUDE_FROM_ALL)
add_library(turbojpeg ALIAS turbojpeg-static)

if (WIN32)
    set(JPEG_TURBO_LIBRARIES turbojpeg-static)  # libturbojpeg.lib
else()
    set(JPEG_TURBO_LIBRARIES turbojpeg)  # libturbojpeg.a
endif()
set(JPEG_TURBO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo
                            ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo)
set(JPEG_TURBO_LIBRARIES ${JPEG_TURBO_LIBRARIES} PARENT_SCOPE)
set(JPEG_TURBO_INCLUDE_DIRS ${JPEG_TURBO_INCLUDE_DIRS} PARENT_SCOPE)

target_include_directories(turbojpeg-static PUBLIC
    ${JPEG_TURBO_INCLUDE_DIRS}
)

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS turbojpeg-static  # libturbojpeg.a will be installed
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
