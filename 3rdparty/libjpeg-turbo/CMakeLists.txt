if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
endif()

set(ENABLE_STATIC ON CACHE BOOL "Force libjpeg to enable static lib" FORCE)

include(CheckLanguage)
check_language(ASM_NASM)
if (CMAKE_ASM_NASM_COMPILER)
    enable_language(ASM_NASM)
    option(WITH_SIMD "" ON)
else()
    message(WARNING "NASM assembler not found - libjpeg-turbo performance may suffer")
    option(WITH_SIMD "" OFF)
endif()

add_subdirectory(libjpeg-turbo EXCLUDE_FROM_ALL)
add_library(turbojpeg ALIAS turbojpeg-static)

set(JPEG_TURBO_LIBRARIES turbojpeg)  # libturbojpeg.a
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
