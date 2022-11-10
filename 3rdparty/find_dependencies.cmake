#
# Open3D 3rd party library integration
#
set(Open3D_3RDPARTY_DIR "${CMAKE_CURRENT_LIST_DIR}")

# EXTERNAL_MODULES
# CMake modules we depend on in our public interface. These are modules we
# need to find_package() in our CMake config script, because we will use their
# targets.
set(Open3D_3RDPARTY_EXTERNAL_MODULES)

# XXX_FROM_CUSTOM vs. XXX_FROM_SYSTEM
# - "FROM_CUSTOM": downloaded or compiled
# - "FROM_SYSTEM": installed with a system package manager, deteced by CMake

# PUBLIC_TARGETS
# CMake targets we link against in our public interface. They are either locally
# defined and installed, or imported from an external module (see above).
set(Open3D_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM)
set(Open3D_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM)

# HEADER_TARGETS
# CMake targets we use in our public interface, but as a special case we only
# need to link privately against the library. This simplifies dependencies
# where we merely expose declared data types from other libraries in our
# public headers, so it would be overkill to require all library users to link
# against that dependency.
set(Open3D_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM)
set(Open3D_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM)

# PRIVATE_TARGETS
# CMake targets for dependencies which are not exposed in the public API. This
# will include anything else we use internally.
set(Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM)
set(Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM)

find_package(PkgConfig QUIET)

# open3d_build_3rdparty_library(name ...)
#
# Builds a third-party library from source
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    INCLUDE_ALL
#        install all files in the include directories. Default is *.h, *.hpp
#    VISIBLE
#        Symbols from this library will be visible for use outside Open3D.
#        Required, for example, if it may throw exceptions that need to be
#        caught in client code.
#    DIRECTORY <dir>
#        the library source directory <dir> is either a subdirectory of
#        3rdparty/ or an absolute directory.
#    INCLUDE_DIRS <dir> [<dir> ...]
#        include headers are in the subdirectories <dir>. Trailing slashes
#        have the same meaning as with install(DIRECTORY). <dir> must be
#        relative to the library source directory.
#        If your include is "#include <x.hpp>" and the path of the file is
#        "path/to/libx/x.hpp" then you need to pass "path/to/libx/"
#        with the trailing "/". If you have "#include <libx/x.hpp>" then you
#        need to pass "path/to/libx".
#    SOURCES <src> [<src> ...]
#        the library sources. Can be omitted for header-only libraries.
#        All sources must be relative to the library source directory.
#    LIBS <target> [<target> ...]
#        extra link dependencies
#    DEPENDS <target> [<target> ...]
#        targets on which <name> depends on and that must be built before.
#
function(open3d_build_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL;VISIBLE" "DIRECTORY" "INCLUDE_DIRS;SOURCES;LIBS;DEPENDS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: open3d_build_3rdparty_library(${name} ${ARGN})")
    endif()
    get_filename_component(arg_DIRECTORY "${arg_DIRECTORY}" ABSOLUTE BASE_DIR "${Open3D_3RDPARTY_DIR}")
    if(arg_SOURCES)
        add_library(${name} STATIC)
        set_target_properties(${name} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_${name}")
        open3d_set_global_properties(${name})
    else()
        add_library(${name} INTERFACE)
    endif()
    if(arg_INCLUDE_DIRS)
        set(include_dirs)
        foreach(incl IN LISTS arg_INCLUDE_DIRS)
            list(APPEND include_dirs "${arg_DIRECTORY}/${incl}")
        endforeach()
    else()
        set(include_dirs "${arg_DIRECTORY}/")
    endif()
    if(arg_SOURCES)
        foreach(src IN LISTS arg_SOURCES)
            get_filename_component(abs_src "${src}" ABSOLUTE BASE_DIR "${arg_DIRECTORY}")
            # Mark as generated to skip CMake's file existence checks
            set_source_files_properties(${abs_src} PROPERTIES GENERATED TRUE)
            target_sources(${name} PRIVATE ${abs_src})
        endforeach()
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM PUBLIC $<BUILD_INTERFACE:${incl_path}>)
        endforeach()
        # Do not export symbols from 3rd party libraries outside the Open3D DSO.
        if(NOT arg_PUBLIC AND NOT arg_HEADER AND NOT arg_VISIBLE)
            set_target_properties(${name} PROPERTIES
                C_VISIBILITY_PRESET hidden
                CXX_VISIBILITY_PRESET hidden
                CUDA_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
            )
        endif()
        if(arg_LIBS)
            target_link_libraries(${name} PRIVATE ${arg_LIBS})
        endif()
    else()
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM INTERFACE $<BUILD_INTERFACE:${incl_path}>)
        endforeach()
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets
            RUNTIME DESTINATION ${Open3D_INSTALL_BIN_DIR}
            ARCHIVE DESTINATION ${Open3D_INSTALL_LIB_DIR}
            LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
        )
    endif()
    if(arg_PUBLIC OR arg_HEADER)
        foreach(incl IN LISTS include_dirs)
            if(arg_INCLUDE_ALL)
                install(DIRECTORY ${incl}
                    DESTINATION ${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                )
            else()
                install(DIRECTORY ${incl}
                    DESTINATION ${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    FILES_MATCHING
                        PATTERN "*.h"
                        PATTERN "*.hpp"
                )
            endif()
            target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty>)
        endforeach()
    endif()
    if(arg_DEPENDS)
        add_dependencies(${name} ${arg_DEPENDS})
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

# CMake arguments for configuring ExternalProjects. Use the second _hidden
# version by default.
set(ExternalProject_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    -DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}
    -DCMAKE_INSTALL_LIBDIR=${Open3D_INSTALL_LIB_DIR}
    # Always build 3rd party code in Release mode. Ignored by multi-config
    # generators (XCode, MSVC). MSVC needs matching config anyway.
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW
    -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
# Keep 3rd party symbols hidden from Open3D user code. Do not use if 3rd party
# libraries throw exceptions that escape Open3D.
set(ExternalProject_CMAKE_ARGS_hidden
    ${ExternalProject_CMAKE_ARGS}
    # Apply LANG_VISIBILITY_PRESET to static libraries and archives as well
    -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_CUDA_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    )

# open3d_pkg_config_3rdparty_library(name ...)
#
# Creates an interface library for a pkg-config dependency.
#
# The function will set ${name}_FOUND to TRUE or FALSE
# indicating whether or not the library could be found.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    SEARCH_ARGS
#        the arguments passed to pkg_search_module()
#
function(open3d_pkg_config_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER" "" "SEARCH_ARGS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: open3d_pkg_config_3rdparty_library(${name} ${ARGN})")
    endif()
    if(PKGCONFIG_FOUND)
        pkg_search_module(pc_${name} ${arg_SEARCH_ARGS})
    endif()
    if(pc_${name}_FOUND)
        message(STATUS "Using installed third-party library ${name} ${${name_uc}_VERSION}")
        add_library(${name} INTERFACE)
        target_include_directories(${name} SYSTEM INTERFACE ${pc_${name}_INCLUDE_DIRS})
        target_link_libraries(${name} INTERFACE ${pc_${name}_LINK_LIBRARIES})
        foreach(flag IN LISTS pc_${name}_CFLAGS_OTHER)
            if(flag MATCHES "-D(.*)")
                target_compile_definitions(${name} INTERFACE ${CMAKE_MATCH_1})
            endif()
        endforeach()
        if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
            install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
        endif()
        set(${name}_FOUND TRUE PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# open3d_find_package_3rdparty_library(name ...)
#
# Creates an interface library for a find_package dependency.
#
# The function will set ${name}_FOUND to TRUE or FALSE
# indicating whether or not the library could be found.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    REQUIRED
#        finding the package is required
#    QUIET
#        finding the package is quiet
#    PACKAGE <pkg>
#        the name of the queried package <pkg> forwarded to find_package()
#    PACKAGE_VERSION_VAR <pkg_version>
#        the variable <pkg_version> where to find the version of the queried package <pkg> find_package().
#        If not provided, PACKAGE_VERSION_VAR will default to <pkg>_VERSION.
#    TARGETS <target> [<target> ...]
#        the expected targets to be found in <pkg>
#    INCLUDE_DIRS
#        the expected include directory variable names to be found in <pkg>.
#        If <pkg> also defines targets, use them instead and pass them via TARGETS option.
#    LIBRARIES
#        the expected library variable names to be found in <pkg>.
#        If <pkg> also defines targets, use them instead and pass them via TARGETS option.
#
function(open3d_find_package_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;REQUIRED;QUIET" "PACKAGE;PACKAGE_VERSION_VAR" "TARGETS;INCLUDE_DIRS;LIBRARIES" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: open3d_find_package_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_PACKAGE)
        message(FATAL_ERROR "open3d_find_package_3rdparty_library: Expected value for argument PACKAGE")
    endif()
    if(NOT arg_PACKAGE_VERSION_VAR)
        set(arg_PACKAGE_VERSION_VAR "${arg_PACKAGE}_VERSION")
    endif()
    set(find_package_args "")
    if(arg_REQUIRED)
        list(APPEND find_package_args "REQUIRED")
    endif()
    if(arg_QUIET)
        list(APPEND find_package_args "QUIET")
    endif()
    find_package(${arg_PACKAGE} ${find_package_args})
    if(${arg_PACKAGE}_FOUND)
        message(STATUS "Using installed third-party library ${name} ${${arg_PACKAGE}_VERSION}")
        add_library(${name} INTERFACE)
        if(arg_TARGETS)
            foreach(target IN LISTS arg_TARGETS)
                if (TARGET ${target})
                    target_link_libraries(${name} INTERFACE ${target})
                else()
                    message(WARNING "Skipping undefined target ${target}")
                endif()
            endforeach()
        endif()
        if(arg_INCLUDE_DIRS)
            foreach(incl IN LISTS arg_INCLUDE_DIRS)
                target_include_directories(${name} INTERFACE ${${incl}})
            endforeach()
        endif()
        if(arg_LIBRARIES)
            foreach(lib IN LISTS arg_LIBRARIES)
                target_link_libraries(${name} INTERFACE ${${lib}})
            endforeach()
        endif()
        if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
            install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
            # Ensure that imported targets will be found again.
            if(arg_TARGETS)
                list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES ${arg_PACKAGE})
                set(Open3D_3RDPARTY_EXTERNAL_MODULES ${Open3D_3RDPARTY_EXTERNAL_MODULES} PARENT_SCOPE)
            endif()
        endif()
        set(${name}_FOUND TRUE PARENT_SCOPE)
        set(${name}_VERSION ${${arg_PACKAGE_VERSION_VAR}} PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# List of linker options for libOpen3D client binaries (eg: pybind) to hide Open3D 3rd
# party dependencies. Only needed with GCC, not AppleClang.
set(OPEN3D_HIDDEN_3RDPARTY_LINK_OPTIONS)

if (CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
    find_library(LexLIB libl.a)    # test archive in macOS
    if (LexLIB)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_LINK_OPTIONS -load_hidden ${LexLIB})
        check_cxx_source_compiles("int main() {return 0;}" FLAG_load_hidden)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
    endif()
endif()
if (NOT FLAG_load_hidden)
    set(FLAG_load_hidden 0)
endif()

# open3d_import_3rdparty_library(name ...)
#
# Imports a third-party library that has been built independently in a sub project.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface and will be
#        installed, but the library is linked privately.
#    INCLUDE_ALL
#        install all files in the include directories. Default is *.h, *.hpp
#    HIDDEN
#        Symbols from this library will not be exported to client code during
#        linking with Open3D. This is the opposite of the VISIBLE option in
#        open3d_build_3rdparty_library.  Prefer hiding symbols during building 3rd
#        party libraries, since this option is not supported by the MSVC linker.
#    GROUPED
#        add "-Wl,--start-group" libx.a liby.a libz.a "-Wl,--end-group" around
#        the libraries.
#    INCLUDE_DIRS
#        the temporary location where the library headers have been installed.
#        Trailing slashes have the same meaning as with install(DIRECTORY).
#        If your include is "#include <x.hpp>" and the path of the file is
#        "/path/to/libx/x.hpp" then you need to pass "/path/to/libx/"
#        with the trailing "/". If you have "#include <libx/x.hpp>" then you
#        need to pass "/path/to/libx".
#    LIBRARIES
#        the built library name(s). It is assumed that the library is static.
#        If the library is PUBLIC, it will be renamed to Open3D_${name} at
#        install time to prevent name collisions in the install space.
#    LIB_DIR
#        the temporary location of the library. Defaults to
#        CMAKE_ARCHIVE_OUTPUT_DIRECTORY.
#    DEPENDS <target> [<target> ...]
#        targets on which <name> depends on and that must be built before.
#
function(open3d_import_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL;HIDDEN;GROUPED" "LIB_DIR" "INCLUDE_DIRS;LIBRARIES;DEPENDS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: open3d_import_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_LIB_DIR)
        set(arg_LIB_DIR "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    endif()
    add_library(${name} INTERFACE)
    if(arg_INCLUDE_DIRS)
        foreach(incl IN LISTS arg_INCLUDE_DIRS)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM INTERFACE $<BUILD_INTERFACE:${incl_path}>)
            if(arg_PUBLIC OR arg_HEADER)
                if(arg_INCLUDE_ALL)
                    install(DIRECTORY ${incl}
                        DESTINATION ${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    )
                else()
                    install(DIRECTORY ${incl}
                        DESTINATION ${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                        FILES_MATCHING
                            PATTERN "*.h"
                            PATTERN "*.hpp"
                    )
                endif()
                target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty>)
            endif()
        endforeach()
    endif()
    if(arg_LIBRARIES)
        list(LENGTH arg_LIBRARIES libcount)
        if(arg_HIDDEN AND NOT arg_PUBLIC AND NOT arg_HEADER)
            set(HIDDEN 1)
        else()
            set(HIDDEN 0)
        endif()
        if(arg_GROUPED)
            target_link_libraries(${name} INTERFACE "-Wl,--start-group")
        endif()
        foreach(arg_LIBRARY IN LISTS arg_LIBRARIES)
            set(library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            if(libcount EQUAL 1)
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}${CMAKE_STATIC_LIBRARY_SUFFIX})
            else()
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}_${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            endif()
            # Apple compiler ld
            target_link_libraries(${name} INTERFACE
                "$<BUILD_INTERFACE:$<$<AND:${HIDDEN},${FLAG_load_hidden}>:-load_hidden >${arg_LIB_DIR}/${library_filename}>")
            if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
                install(FILES ${arg_LIB_DIR}/${library_filename}
                    DESTINATION ${Open3D_INSTALL_LIB_DIR}
                    RENAME ${installed_library_filename}
                )
                target_link_libraries(${name} INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${Open3D_INSTALL_LIB_DIR}/${installed_library_filename}>)
            endif()
            if (HIDDEN)
                # GNU compiler ld
                target_link_options(${name} INTERFACE
                    $<$<CXX_COMPILER_ID:GNU>:LINKER:--exclude-libs,${library_filename}>)
                list(APPEND OPEN3D_HIDDEN_3RDPARTY_LINK_OPTIONS $<$<CXX_COMPILER_ID:GNU>:LINKER:--exclude-libs,${library_filename}>)
                set(OPEN3D_HIDDEN_3RDPARTY_LINK_OPTIONS
                    ${OPEN3D_HIDDEN_3RDPARTY_LINK_OPTIONS} PARENT_SCOPE)
            endif()
        endforeach()
        if(arg_GROUPED)
            target_link_libraries(${name} INTERFACE "-Wl,--end-group")
        endif()
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
    endif()
    if(arg_DEPENDS)
        add_dependencies(${name} ${arg_DEPENDS})
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

include(ProcessorCount)
ProcessorCount(NPROC)

# CUDAToolkit (required at this point for subsequent checks and targets)
if(BUILD_CUDA_MODULE)
    find_package(CUDAToolkit REQUIRED)
endif()

# Threads
open3d_find_package_3rdparty_library(3rdparty_threads
    REQUIRED
    PACKAGE Threads
    TARGETS Threads::Threads
)

# OpenMP
if(WITH_OPENMP)
    open3d_find_package_3rdparty_library(3rdparty_openmp
        PACKAGE OpenMP
        PACKAGE_VERSION_VAR OpenMP_CXX_VERSION
        TARGETS OpenMP::OpenMP_CXX
    )
    if(3rdparty_openmp_FOUND)
        message(STATUS "Building with OpenMP")
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_openmp)
    endif()
endif()

# X11
if(UNIX AND NOT APPLE)
    open3d_find_package_3rdparty_library(3rdparty_x11
        QUIET
        PACKAGE X11
        TARGETS X11::X11
    )
endif()

# CUB (already included in CUDA 11.0+)
if(BUILD_CUDA_MODULE AND CUDAToolkit_VERSION VERSION_LESS "11.0")
    include(${Open3D_3RDPARTY_DIR}/cub/cub.cmake)
    open3d_import_3rdparty_library(3rdparty_cub
        INCLUDE_DIRS ${CUB_INCLUDE_DIRS}
        DEPENDS      ext_cub
    )
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_cub)
endif()

# cutlass
if(BUILD_CUDA_MODULE)
    include(${Open3D_3RDPARTY_DIR}/cutlass/cutlass.cmake)
    open3d_import_3rdparty_library(3rdparty_cutlass
        INCLUDE_DIRS ${CUTLASS_INCLUDE_DIRS}
        DEPENDS      ext_cutlass
    )
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_cutlass)
endif()

# Dirent
if(WIN32)
    open3d_build_3rdparty_library(3rdparty_dirent DIRECTORY dirent)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_dirent)
endif()

# Eigen3
if(USE_SYSTEM_EIGEN3)
    open3d_find_package_3rdparty_library(3rdparty_eigen3
        PUBLIC
        PACKAGE Eigen3
        TARGETS Eigen3::Eigen
    )
    if(NOT 3rdparty_eigen3_FOUND)
        set(USE_SYSTEM_EIGEN3 OFF)
    endif()
endif()
if(NOT USE_SYSTEM_EIGEN3)
    include(${Open3D_3RDPARTY_DIR}/eigen/eigen.cmake)
    open3d_import_3rdparty_library(3rdparty_eigen3
        PUBLIC
        INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS}
        INCLUDE_ALL
        DEPENDS      ext_eigen
    )
    list(APPEND Open3D_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM Open3D::3rdparty_eigen3)
else()
    list(APPEND Open3D_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM Open3D::3rdparty_eigen3)
endif()

# Nanoflann
if(USE_SYSTEM_NANOFLANN)
    open3d_find_package_3rdparty_library(3rdparty_nanoflann
        PACKAGE nanoflann
        TARGETS nanoflann::nanoflann
    )
    if(NOT 3rdparty_nanoflann_FOUND)
        set(USE_SYSTEM_NANOFLANN OFF)
    endif()
endif()
if(NOT USE_SYSTEM_NANOFLANN)
    include(${Open3D_3RDPARTY_DIR}/nanoflann/nanoflann.cmake)
    open3d_import_3rdparty_library(3rdparty_nanoflann
        INCLUDE_DIRS ${NANOFLANN_INCLUDE_DIRS}
        DEPENDS      ext_nanoflann
    )
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_nanoflann)
else()
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_nanoflann)
endif()

# TurboJPEG
if(USE_SYSTEM_JPEG AND BUILD_AZURE_KINECT)
    open3d_pkg_config_3rdparty_library(3rdparty_turbojpeg
        SEARCH_ARGS turbojpeg
    )
    if(NOT 3rdparty_turbojpeg_FOUND)
        message(STATUS "Azure Kinect driver needs TurboJPEG API")
        set(USE_SYSTEM_JPEG OFF)
    endif()
endif()

# JPEG
if(USE_SYSTEM_JPEG)
    open3d_find_package_3rdparty_library(3rdparty_jpeg
        PACKAGE JPEG
        TARGETS JPEG::JPEG
    )
    if(3rdparty_jpeg_FOUND)
        if(TARGET Open3D::3rdparty_turbojpeg)
            list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_turbojpeg)
        endif()
    else()
        set(USE_SYSTEM_JPEG OFF)
    endif()
endif()
if(NOT USE_SYSTEM_JPEG)
    message(STATUS "Building third-party library JPEG from source")
    include(${Open3D_3RDPARTY_DIR}/libjpeg-turbo/libjpeg-turbo.cmake)
    open3d_import_3rdparty_library(3rdparty_jpeg
        INCLUDE_DIRS ${JPEG_TURBO_INCLUDE_DIRS}
        LIB_DIR      ${JPEG_TURBO_LIB_DIR}
        LIBRARIES    ${JPEG_TURBO_LIBRARIES}
        DEPENDS      ext_turbojpeg
    )
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_jpeg)
else()
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_jpeg)
endif()

# jsoncpp
if(USE_SYSTEM_JSONCPP)
    open3d_find_package_3rdparty_library(3rdparty_jsoncpp
        PACKAGE jsoncpp
        TARGETS jsoncpp_lib
    )
    if(NOT 3rdparty_jsoncpp_FOUND)
        set(USE_SYSTEM_JSONCPP OFF)
    endif()
endif()
if(NOT USE_SYSTEM_JSONCPP)
    include(${Open3D_3RDPARTY_DIR}/jsoncpp/jsoncpp.cmake)
    open3d_import_3rdparty_library(3rdparty_jsoncpp
        INCLUDE_DIRS ${JSONCPP_INCLUDE_DIRS}
        LIB_DIR      ${JSONCPP_LIB_DIR}
        LIBRARIES    ${JSONCPP_LIBRARIES}
        DEPENDS      ext_jsoncpp
    )
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_jsoncpp)
else()
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_jsoncpp)
endif()

# liblzf
if(USE_SYSTEM_LIBLZF)
    open3d_find_package_3rdparty_library(3rdparty_liblzf
        PACKAGE liblzf
        TARGETS liblzf::liblzf
    )
    if(NOT 3rdparty_liblzf_FOUND)
        set(USE_SYSTEM_LIBLZF OFF)
    endif()
endif()
if(NOT USE_SYSTEM_LIBLZF)
    open3d_build_3rdparty_library(3rdparty_liblzf DIRECTORY liblzf
        SOURCES
            liblzf/lzf_c.c
            liblzf/lzf_d.c
    )
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_liblzf)
else()
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_liblzf)
endif()

# tritriintersect
open3d_build_3rdparty_library(3rdparty_tritriintersect DIRECTORY tomasakeninemoeller
    INCLUDE_DIRS include/
)
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_tritriintersect)

# Curl
# - Curl should be linked before PNG, otherwise it will have undefined symbols.
# - openssl.cmake needs to be included before curl.cmake, for the
#   BORINGSSL_ROOT_DIR variable.
include(${Open3D_3RDPARTY_DIR}/boringssl/boringssl.cmake)
include(${Open3D_3RDPARTY_DIR}/curl/curl.cmake)
open3d_import_3rdparty_library(3rdparty_curl
    INCLUDE_DIRS ${CURL_INCLUDE_DIRS}
    INCLUDE_ALL
    LIB_DIR      ${CURL_LIB_DIR}
    LIBRARIES    ${CURL_LIBRARIES}
    DEPENDS      ext_zlib ext_curl
)
if(APPLE)
    # Missing frameworks: https://stackoverflow.com/a/56157695/1255535
    # Link frameworks   : https://stackoverflow.com/a/18330634/1255535
    # Fixes error:
    # ```
    # Undefined symbols for architecture arm64:
    # "_SCDynamicStoreCopyProxies", referenced from:
    #     _Curl_resolv in libcurl.a(hostip.c.o)
    # ```
    # The "Foundation" framework is already linked by GLFW.
    target_link_libraries(3rdparty_curl INTERFACE "-framework SystemConfiguration")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_curl)

# BoringSSL
open3d_import_3rdparty_library(3rdparty_openssl
    INCLUDE_DIRS ${BORINGSSL_INCLUDE_DIRS}
    INCLUDE_ALL
    INCLUDE_DIRS ${BORINGSSL_INCLUDE_DIRS}
    LIB_DIR      ${BORINGSSL_LIB_DIR}
    LIBRARIES    ${BORINGSSL_LIBRARIES}
    DEPENDS      ext_zlib ext_boringssl
)
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_openssl)

# PNG
if(USE_SYSTEM_PNG)
    # ZLIB::ZLIB is automatically included by the PNG package.
    open3d_find_package_3rdparty_library(3rdparty_png
        PACKAGE PNG
        PACKAGE_VERSION_VAR PNG_VERSION_STRING
        TARGETS PNG::PNG
    )
    if(NOT 3rdparty_png_FOUND)
        set(USE_SYSTEM_PNG OFF)
    endif()
endif()
if(NOT USE_SYSTEM_PNG)
    include(${Open3D_3RDPARTY_DIR}/zlib/zlib.cmake)
    open3d_import_3rdparty_library(3rdparty_zlib
        HIDDEN
        INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS}
        LIB_DIR      ${ZLIB_LIB_DIR}
        LIBRARIES    ${ZLIB_LIBRARIES}
        DEPENDS      ext_zlib
    )

    include(${Open3D_3RDPARTY_DIR}/libpng/libpng.cmake)
    open3d_import_3rdparty_library(3rdparty_png
        INCLUDE_DIRS ${LIBPNG_INCLUDE_DIRS}
        LIB_DIR      ${LIBPNG_LIB_DIR}
        LIBRARIES    ${LIBPNG_LIBRARIES}
        DEPENDS      ext_libpng
    )
    add_dependencies(ext_libpng ext_zlib)
    target_link_libraries(3rdparty_png INTERFACE Open3D::3rdparty_zlib)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_png)
else()
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_png)
endif()

# fmt
if(USE_SYSTEM_FMT)
    open3d_find_package_3rdparty_library(3rdparty_fmt
        PUBLIC
        PACKAGE fmt
        TARGETS fmt::fmt
    )
    if(NOT 3rdparty_fmt_FOUND)
        set(USE_SYSTEM_FMT OFF)
    endif()
endif()
if(NOT USE_SYSTEM_FMT)
    include(${Open3D_3RDPARTY_DIR}/fmt/fmt.cmake)
    open3d_import_3rdparty_library(3rdparty_fmt
        HEADER
        INCLUDE_DIRS ${FMT_INCLUDE_DIRS}
        LIB_DIR      ${FMT_LIB_DIR}
        LIBRARIES    ${FMT_LIBRARIES}
        DEPENDS      ext_fmt
    )
    # FMT 6.0, newer versions may require different flags
    target_compile_definitions(3rdparty_fmt INTERFACE FMT_HEADER_ONLY=0)
    target_compile_definitions(3rdparty_fmt INTERFACE FMT_USE_WINDOWS_H=0)
    target_compile_definitions(3rdparty_fmt INTERFACE FMT_STRING_ALIAS=1)
    list(APPEND Open3D_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM Open3D::3rdparty_fmt)
else()
    list(APPEND Open3D_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM Open3D::3rdparty_fmt)
endif()

# Pybind11
if (BUILD_PYTHON_MODULE)
    if(USE_SYSTEM_PYBIND11)
        find_package(pybind11)
    endif()
    if (NOT USE_SYSTEM_PYBIND11 OR NOT TARGET pybind11::module)
        set(USE_SYSTEM_PYBIND11 OFF)
        include(${Open3D_3RDPARTY_DIR}/pybind11/pybind11.cmake)
        # pybind11 will automatically become available.
    endif()
endif()

# Googletest
if (BUILD_UNIT_TESTS)
    if(USE_SYSTEM_GOOGLETEST)
        open3d_find_package_3rdparty_library(3rdparty_googletest
            PACKAGE GTest
            TARGETS GTest::gmock
        )
        if(NOT 3rdparty_googletest_FOUND)
            set(USE_SYSTEM_GOOGLETEST OFF)
        endif()
    endif()
    if(NOT USE_SYSTEM_GOOGLETEST)
        include(${Open3D_3RDPARTY_DIR}/googletest/googletest.cmake)
        open3d_build_3rdparty_library(3rdparty_googletest DIRECTORY ${GOOGLETEST_SOURCE_DIR}
            SOURCES
                googletest/src/gtest-all.cc
                googlemock/src/gmock-all.cc
            INCLUDE_DIRS
                googletest/include/
                googletest/
                googlemock/include/
                googlemock/
            DEPENDS
                ext_googletest
        )
    endif()
endif()

# Google benchmark
if (BUILD_BENCHMARKS)
    include(${Open3D_3RDPARTY_DIR}/benchmark/benchmark.cmake)
    # benchmark and benchmark_main will automatically become available.
endif()

if(BUILD_SYCL_MODULE)
    add_library(3rdparty_sycl INTERFACE)
    target_link_libraries(3rdparty_sycl INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<LINK_LANGUAGE:ISPC>>>:sycl>)
    target_link_options(3rdparty_sycl INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<LINK_LANGUAGE:ISPC>>>:-fsycl -fsycl-targets=spir64_x86_64>)
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS 3rdparty_sycl EXPORT Open3DTargets)
    endif()
    add_library(Open3D::3rdparty_sycl ALIAS 3rdparty_sycl)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_sycl)
endif()

if(BUILD_SYCL_MODULE)
    option(OPEN3D_USE_ONEAPI_PACKAGES "Use the oneAPI distribution of MKL/TBB/DPL." ON)
else()
    option(OPEN3D_USE_ONEAPI_PACKAGES "Use the oneAPI distribution of MKL/TBB/DPL." OFF)
endif()
mark_as_advanced(OPEN3D_USE_ONEAPI_PACKAGES)

if(OPEN3D_USE_ONEAPI_PACKAGES)
    # 1. oneTBB
    # /opt/intel/oneapi/tbb/latest/lib/cmake/tbb
    open3d_find_package_3rdparty_library(3rdparty_tbb
        PACKAGE TBB
        TARGETS TBB::tbb
    )
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_tbb)

    # 2. oneDPL
    # /opt/intel/oneapi/dpl/latest/lib/cmake/oneDPL
    open3d_find_package_3rdparty_library(3rdparty_onedpl
        PACKAGE oneDPL
        TARGETS oneDPL
    )
    target_compile_definitions(3rdparty_onedpl INTERFACE _GLIBCXX_USE_TBB_PAR_BACKEND=0)
    target_compile_definitions(3rdparty_onedpl INTERFACE PSTL_USE_PARALLEL_POLICIES=0)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_onedpl)

    # 3. oneMKL
    # /opt/intel/oneapi/mkl/latest/lib/cmake/mkl
    set(MKL_THREADING tbb_thread)
    set(MKL_LINK static)
    find_package(MKL REQUIRED)
    open3d_import_3rdparty_library(3rdparty_mkl
        HIDDEN
        GROUPED
        INCLUDE_DIRS ${MKL_INCLUDE}/
        LIB_DIR      ${MKL_ROOT}/lib/intel64
        LIBRARIES    mkl_intel_ilp64 mkl_tbb_thread mkl_core
    )
    # MKL definitions
    target_compile_options(3rdparty_mkl INTERFACE "$<$<PLATFORM_ID:Linux,Darwin>:$<$<COMPILE_LANGUAGE:CXX>:-m64>>")
    target_compile_definitions(3rdparty_mkl INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:MKL_ILP64>")
    # Other global macros
    target_compile_definitions(3rdparty_mkl INTERFACE OPEN3D_USE_ONEAPI_PACKAGES)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_mkl)

else() # if(OPEN3D_USE_ONEAPI_PACKAGES)
    # TBB
    if(USE_SYSTEM_TBB)
        open3d_find_package_3rdparty_library(3rdparty_tbb
            PACKAGE TBB
            TARGETS TBB::tbb
        )
        if(NOT 3rdparty_tbb_FOUND)
            set(USE_SYSTEM_TBB OFF)
        endif()
    endif()
    if(NOT USE_SYSTEM_TBB)
        include(${Open3D_3RDPARTY_DIR}/mkl/tbb.cmake)
        open3d_import_3rdparty_library(3rdparty_tbb
            INCLUDE_DIRS ${STATIC_TBB_INCLUDE_DIR}
            LIB_DIR      ${STATIC_TBB_LIB_DIR}
            LIBRARIES    ${STATIC_TBB_LIBRARIES}
            DEPENDS      ext_tbb
        )
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_tbb)
    else()
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_tbb)
    endif()

    # parallelstl
    include(${Open3D_3RDPARTY_DIR}/parallelstl/parallelstl.cmake)
    open3d_import_3rdparty_library(3rdparty_parallelstl
    PUBLIC
    INCLUDE_DIRS ${PARALLELSTL_INCLUDE_DIRS}
    INCLUDE_ALL
    DEPENDS      ext_parallelstl
    )
    list(APPEND Open3D_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM Open3D::3rdparty_parallelstl)

    # MKL/BLAS
    if(USE_BLAS)
        if (USE_SYSTEM_BLAS)
            find_package(BLAS)
            find_package(LAPACK)
            find_package(LAPACKE)
            if(BLAS_FOUND AND LAPACK_FOUND AND LAPACKE_FOUND)
                message(STATUS "System BLAS/LAPACK/LAPACKE found.")
                list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM
                    ${BLAS_LIBRARIES}
                    ${LAPACK_LIBRARIES}
                    ${LAPACKE_LIBRARIES}
                )
            else()
                message(STATUS "System BLAS/LAPACK/LAPACKE not found, setting USE_SYSTEM_BLAS=OFF.")
                set(USE_SYSTEM_BLAS OFF)
            endif()
        endif()

        if (NOT USE_SYSTEM_BLAS)
            # Install gfortran first for compiling OpenBLAS/Lapack from source.
            message(STATUS "Building OpenBLAS with LAPACK from source")

            find_program(gfortran_bin "gfortran")
            if (gfortran_bin)
                message(STATUS "gfortran found at ${gfortran}")
            else()
                message(FATAL_ERROR "gfortran is required to compile LAPACK from source. "
                                    "On Ubuntu, please install by `apt install gfortran`. "
                                    "On macOS, please install by `brew install gfortran`. ")
            endif()

            include(${Open3D_3RDPARTY_DIR}/openblas/openblas.cmake)
            open3d_import_3rdparty_library(3rdparty_blas
                HIDDEN
                INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR}
                LIB_DIR      ${OPENBLAS_LIB_DIR}
                LIBRARIES    ${OPENBLAS_LIBRARIES}
                DEPENDS      ext_openblas
            )
            # Get gfortran library search directories.
            execute_process(COMMAND ${gfortran_bin} -print-search-dirs
                OUTPUT_VARIABLE gfortran_search_dirs
                RESULT_VARIABLE RET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(RET AND NOT RET EQUAL 0)
                message(FATAL_ERROR "Failed to run `${gfortran_bin} -print-search-dirs`")
            endif()

            # Parse gfortran library search directories into CMake list.
            string(REGEX MATCH "libraries: =(.*)" match_result ${gfortran_search_dirs})
            if (match_result)
                string(REPLACE ":" ";" gfortran_lib_dirs ${CMAKE_MATCH_1})
            else()
                message(FATAL_ERROR "Failed to parse gfortran_search_dirs: ${gfortran_search_dirs}")
            endif()

            if(LINUX_AARCH64 OR APPLE_AARCH64)
                # Find libgfortran.a and libgcc.a inside the gfortran library search
                # directories. This ensures that the library matches the compiler.
                # On ARM64 Ubuntu and ARM64 macOS, libgfortran.a is compiled with `-fPIC`.
                find_library(gfortran_lib NAMES libgfortran.a PATHS ${gfortran_lib_dirs} REQUIRED)
                find_library(gcc_lib      NAMES libgcc.a      PATHS ${gfortran_lib_dirs} REQUIRED)
                find_library(quadmath_lib NAMES libquadmath.a PATHS ${gfortran_lib_dirs} REQUIRED)
                target_link_libraries(3rdparty_blas INTERFACE
                    ${gfortran_lib}
                    ${gcc_lib}
                    ${quadmath_lib}
                )
                if(APPLE_AARCH64)
                    # Suppress Apple compiler warnigns.
                    target_link_options(3rdparty_blas INTERFACE "-Wl,-no_compact_unwind")
                endif()
            elseif(UNIX AND NOT APPLE)
                # On Ubuntu 20.04 x86-64, libgfortran.a is not compiled with `-fPIC`.
                # The temporary solution is to link the shared library libgfortran.so.
                # If we distribute a Python wheel, the user's system will also need
                # to have libgfortran.so preinstalled.
                #
                # If you have to link libgfortran.a statically
                # - Read https://gcc.gnu.org/wiki/InstallingGCC
                # - Run `gfortran --version`, e.g. you get 9.3.0
                # - Checkout gcc source code to the corresponding version
                # - Configure with
                #   ${PWD}/../gcc/configure --prefix=${HOME}/gcc-9.3.0 \
                #                           --enable-languages=c,c++,fortran \
                #                           --with-pic --disable-multilib
                # - make install -j$(nproc) # This will take a while
                # - Change this cmake file to libgfortran.a statically.
                # - Link
                #   - libgfortran.a
                #   - libgcc.a
                #   - libquadmath.a
                target_link_libraries(3rdparty_blas INTERFACE gfortran)
            endif()
            list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_blas)
        endif()
    else()
        include(${Open3D_3RDPARTY_DIR}/mkl/mkl.cmake)
        # MKL, cuSOLVER, cuBLAS
        # We link MKL statically. For MKL link flags, refer to:
        # https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html
        message(STATUS "Using MKL to support BLAS and LAPACK functionalities.")
        open3d_import_3rdparty_library(3rdparty_blas
            HIDDEN
            INCLUDE_DIRS ${STATIC_MKL_INCLUDE_DIR}
            LIB_DIR      ${STATIC_MKL_LIB_DIR}
            LIBRARIES    ${STATIC_MKL_LIBRARIES}
            DEPENDS      ext_tbb ext_mkl_include ext_mkl
        )
        if(UNIX)
            target_compile_options(3rdparty_blas INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-m64>")
            target_link_libraries(3rdparty_blas INTERFACE Open3D::3rdparty_threads ${CMAKE_DL_LIBS})
        endif()
        target_compile_definitions(3rdparty_blas INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:MKL_ILP64>")
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_blas)
    endif()
endif() # if(OPEN3D_USE_ONEAPI_PACKAGES)

# cuBLAS
if(BUILD_CUDA_MODULE)
    if(WIN32)
        # Nvidia does not provide static libraries for Windows. We don't release
        # pip wheels for Windows with CUDA support at the moment. For the pip
        # wheels to support CUDA on Windows out-of-the-box, we need to either
        # ship the CUDA toolkit with the wheel (e.g. PyTorch can make use of the
        # cudatoolkit conda package), or have a mechanism to locate the CUDA
        # toolkit from the system.
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM CUDA::cusolver CUDA::cublas)
    else()
        # CMake docs   : https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
        # cusolver 11.0: https://docs.nvidia.com/cuda/archive/11.0/cusolver/index.html#static-link-lapack
        # cublas   11.0: https://docs.nvidia.com/cuda/archive/11.0/cublas/index.html#static-library
        # The link order below is important. Theoretically we should use
        # open3d_find_package_3rdparty_library, but we have to insert
        # liblapack_static.a in the middle of the targets.
        add_library(3rdparty_cublas INTERFACE)
        target_link_libraries(3rdparty_cublas INTERFACE
            CUDA::cusolver_static
            ${CUDAToolkit_LIBRARY_DIR}/liblapack_static.a
            CUDA::cusparse_static
            CUDA::cublas_static
            CUDA::cublasLt_static
            CUDA::culibos
        )
        if(NOT BUILD_SHARED_LIBS)
            # Listed in ${CMAKE_INSTALL_PREFIX}/lib/cmake/Open3D/Open3DTargets.cmake.
            install(TARGETS 3rdparty_cublas EXPORT Open3DTargets)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "CUDAToolkit")
        endif()
        add_library(Open3D::3rdparty_cublas ALIAS 3rdparty_cublas)
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_cublas)
    endif()
endif()

# NPP
if (BUILD_CUDA_MODULE)
    # NPP library list: https://docs.nvidia.com/cuda/npp/index.html
    if(WIN32)
        open3d_find_package_3rdparty_library(3rdparty_cuda_npp
            REQUIRED
            PACKAGE CUDAToolkit
            TARGETS CUDA::nppc
                    CUDA::nppicc
                    CUDA::nppif
                    CUDA::nppig
                    CUDA::nppim
                    CUDA::nppial
        )
    else()
        open3d_find_package_3rdparty_library(3rdparty_cuda_npp
            REQUIRED
            PACKAGE CUDAToolkit
            TARGETS CUDA::nppc_static
                    CUDA::nppicc_static
                    CUDA::nppif_static
                    CUDA::nppig_static
                    CUDA::nppim_static
                    CUDA::nppial_static
        )
    endif()
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_cuda_npp)
endif ()

# IPP
if (WITH_IPPICV)
    # Ref: https://stackoverflow.com/a/45125525
    set(IPPICV_SUPPORTED_HW AMD64 x86_64 x64 x86 X86 i386 i686)
    # Unsupported: ARM64 aarch64 armv7l armv8b armv8l ...
    if (NOT CMAKE_HOST_SYSTEM_PROCESSOR IN_LIST IPPICV_SUPPORTED_HW)
        set(WITH_IPPICV OFF)
        message(WARNING "IPP-ICV disabled: Unsupported Platform.")
    else ()
        include(${Open3D_3RDPARTY_DIR}/ippicv/ippicv.cmake)
        if (WITH_IPPICV)
            message(STATUS "IPP-ICV ${IPPICV_VERSION_STRING} available. Building interface wrappers IPP-IW.")
            open3d_import_3rdparty_library(3rdparty_ippicv
                HIDDEN
                INCLUDE_DIRS ${IPPICV_INCLUDE_DIR}
                LIBRARIES    ${IPPICV_LIBRARIES}
                LIB_DIR      ${IPPICV_LIB_DIR}
                DEPENDS      ext_ippicv
            )
            target_compile_definitions(3rdparty_ippicv INTERFACE ${IPPICV_DEFINITIONS})
            list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM Open3D::3rdparty_ippicv)
        endif()
    endif()
endif ()

# Stdgpu
if (BUILD_CUDA_MODULE)
    include(${Open3D_3RDPARTY_DIR}/stdgpu/stdgpu.cmake)
    open3d_import_3rdparty_library(3rdparty_stdgpu
        INCLUDE_DIRS ${STDGPU_INCLUDE_DIRS}
        LIB_DIR      ${STDGPU_LIB_DIR}
        LIBRARIES    ${STDGPU_LIBRARIES}
        DEPENDS      ext_stdgpu
    )
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_stdgpu)
endif ()

# embree
include(${Open3D_3RDPARTY_DIR}/embree/embree.cmake)
open3d_import_3rdparty_library(3rdparty_embree
    HIDDEN
    INCLUDE_DIRS ${EMBREE_INCLUDE_DIRS}
    LIB_DIR      ${EMBREE_LIB_DIR}
    LIBRARIES    ${EMBREE_LIBRARIES}
    DEPENDS      ext_embree
)
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM Open3D::3rdparty_embree)

# Compactify list of external modules.
# This must be called after all dependencies are processed.
list(REMOVE_DUPLICATES Open3D_3RDPARTY_EXTERNAL_MODULES)

# Target linking order matters. When linking a target, the link libraries and
# include directories are set. If the order is wrong, it can cause missing
# symbols or wrong header versions. Generally, we want to link custom libs
# before system libs; the order among differnt libs can also matter.
#
# Current rules:
# 1. 3rdparty_curl should be linked before 3rdparty_png.
# 2. Link "FROM_CUSTOM" before "FROM_SYSTEM" targets.
# 3. ...
set(Open3D_3RDPARTY_PUBLIC_TARGETS
    ${Open3D_3RDPARTY_PUBLIC_TARGETS_FROM_CUSTOM}
    ${Open3D_3RDPARTY_PUBLIC_TARGETS_FROM_SYSTEM})
set(Open3D_3RDPARTY_HEADER_TARGETS
    ${Open3D_3RDPARTY_HEADER_TARGETS_FROM_CUSTOM}
    ${Open3D_3RDPARTY_HEADER_TARGETS_FROM_SYSTEM})
set(Open3D_3RDPARTY_PRIVATE_TARGETS
    ${Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_CUSTOM}
    ${Open3D_3RDPARTY_PRIVATE_TARGETS_FROM_SYSTEM})
