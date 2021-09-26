cmake_minimum_required(VERSION 3.19.2)

project(ispc_isas
    LANGUAGES C
)

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Required by ISPC language emulation
set(BUILD_ISPC_MODULE ON)

# ISPC language emulation support
include(Open3DISPC)

# Build ISPC module by default if ISPC is available
if (CMAKE_ISPC_COMPILER_LOADED OR (CMAKE_GENERATOR MATCHES "Make" OR CMAKE_GENERATOR MATCHES "Ninja"))
    option(ISPC_USE_LEGACY_EMULATION "Use legacy ISPC language emulation over first-class CMake support" OFF)
else()
    option(ISPC_USE_LEGACY_EMULATION "Use legacy ISPC language emulation over first-class CMake support" ON)
endif()
mark_as_advanced(ISPC_USE_LEGACY_EMULATION)
option(ISPC_PRINT_LEGACY_COMPILE_COMMANDS "Prints legacy compile commands on CMake configuration time" ON)
mark_as_advanced(ISPC_PRINT_LEGACY_COMPILE_COMMANDS)

open3d_ispc_enable_language(ISPC)

if (CMAKE_ISPC_COMPILER_ID STREQUAL "Intel" AND CMAKE_ISPC_COMPILER_VERSION VERSION_LESS "1.16")
    message(FATAL_ERROR "ISPC 1.15 and older are not supported. Please upgrade to ISPC 1.16 or newer.")
endif()

if (NOT CMAKE_ISPC_COMPILER_ID)
    message(FATAL_ERROR "Unknown ISPC compiler.")
endif()


open3d_ispc_add_executable(ispc_isas)

target_sources(ispc_isas PRIVATE
    ISADispatcher.c
    ISADispatcher.ispc
)

# Required for including "open3d/utility/ISAInfo.isph"
target_include_directories(ispc_isas PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../cpp"
)
