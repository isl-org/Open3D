#
# Open3D 3rd party library integration
#
set(Open3D_3RDPARTY_DIR "${PROJECT_SOURCE_DIR}/3rdparty")

# EXTERNAL_MODULES
# CMake modules we depend on in our public interface. These are modules we
# need to find_package() in our CMake config script, because we will use their
# targets.
set(Open3D_3RDPARTY_EXTERNAL_MODULES)

# PUBLIC_TARGETS
# CMake targets we link against in our public interface. They are
# either locally defined and installed, or imported from an external module
# (see above).
set(Open3D_3RDPARTY_PUBLIC_TARGETS)

# HEADER_TARGETS
# CMake targets we use in our public interface, but as a special case we do not
# need to link against the library. This simplifies dependencies where we merely
# expose declared data types from other libraries in our public headers, so it
# would be overkill to require all library users to link against that dependency.
set(Open3D_3RDPARTY_HEADER_TARGETS)

# PRIVATE_TARGETS
# CMake targets for dependencies which are not exposed in the public API. This
# will probably include HEADER_TARGETS, but also anything else we use internally.
set(Open3D_3RDPARTY_PRIVATE_TARGETS)

find_package(PkgConfig QUIET)

#
# build_3rdparty_library(name ...)
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
#    DIRECTORY <dir>
#        the library sources are in the subdirectory <dir> of 3rdparty/
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
#
function(build_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL" "DIRECTORY" "INCLUDE_DIRS;SOURCES;LIBS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Invalid syntax: build_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_DIRECTORY)
        set(arg_DIRECTORY "${name}")
    endif()
    if(arg_INCLUDE_DIRS)
        set(include_dirs)
        foreach(incl IN LISTS arg_INCLUDE_DIRS)
            list(APPEND include_dirs "${Open3D_3RDPARTY_DIR}/${arg_DIRECTORY}/${incl}")
        endforeach()
    else()
        set(include_dirs "${Open3D_3RDPARTY_DIR}/${arg_DIRECTORY}/")
    endif()
    message(STATUS "Building library ${name} from source")
    if(arg_SOURCES)
        set(sources)
        foreach(src ${arg_SOURCES})
            list(APPEND sources "${Open3D_3RDPARTY_DIR}/${arg_DIRECTORY}/${src}")
        endforeach()
        add_library(${name} STATIC ${sources})
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM PUBLIC
                $<BUILD_INTERFACE:${incl_path}>
            )
        endforeach()
        target_include_directories(${name} PUBLIC
            $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty>
        )
        open3d_set_global_properties(${name})
        set_target_properties(${name} PROPERTIES
            OUTPUT_NAME "${PROJECT_NAME}_${name}"
        )
        if(arg_LIBS)
            target_link_libraries(${name} PRIVATE ${arg_LIBS})
        endif()
    else()
        add_library(${name} INTERFACE)
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM INTERFACE
                $<BUILD_INTERFACE:${incl_path}>
            )
        endforeach()
        target_include_directories(${name} INTERFACE
            $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty>
        )
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
        endforeach()
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

#
# pkg_config_3rdparty_library(name ...)
#
# Creates an interface library for a pkg-config dependency.
# All arguments are passed verbatim to pkg_search_module()
#
# The function will set ${name}_FOUND to TRUE or FALSE
# indicating whether or not the library could be found.
#
function(pkg_config_3rdparty_library name)
    if(PKGCONFIG_FOUND)
        pkg_search_module(pc_${name} ${ARGN})
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
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
        set(${name}_FOUND TRUE PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

#
# import_3rdparty_library(name ...)
#
# Imports a third-party library that has been built independently in a sub project.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface and will be
#        installed, but the library is linked privately.
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
#
function(import_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER" "LIB_DIR" "INCLUDE_DIRS;LIBRARIES" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: import_3rdparty_library(${name} ${ARGN})")
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
                install(DIRECTORY ${incl} DESTINATION ${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
                )
                target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}/open3d/3rdparty>)
            endif()
        endforeach()
    endif()
    if(arg_LIBRARIES)
        list(LENGTH arg_LIBRARIES libcount)
        foreach(arg_LIBRARY IN LISTS arg_LIBRARIES)
            set(library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            if(libcount EQUAL 1)
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}${CMAKE_STATIC_LIBRARY_SUFFIX})
            else()
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}_${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            endif()
            target_link_libraries(${name} INTERFACE $<BUILD_INTERFACE:${arg_LIB_DIR}/${library_filename}>)
            if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
                install(FILES ${arg_LIB_DIR}/${library_filename}
                    DESTINATION ${Open3D_INSTALL_LIB_DIR}
                    RENAME ${installed_library_filename}
                )
                target_link_libraries(${name} INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${Open3D_INSTALL_LIB_DIR}/${installed_library_filename}>)
            endif()
        endforeach()
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

#
# set_local_or_remote_url(url ...)
#
# If LOCAL_URL exists, set URL to LOCAL_URL, otherwise set URL to REMOTE_URLS.
# This function is needed since CMake does not allow specifying remote URL(s) if
# a local URL is specified.
#
# Valid options:
#    LOCAL_URL
#        local url to a file. Optional parameter. If the file does not exist,
#        LOCAL_URL will be ignored. If the file exists, REMOTE URLS will be
#        ignored. CMake only allows setting single LOCAL_URL for external
#        projects.
#    REMOTE_URLS
#        remote url(s) to download a file. CMake will try to download the file
#        in the specified order.
#
function(set_local_or_remote_url URL)
    cmake_parse_arguments(arg "" "LOCAL_URL" "REMOTE_URLS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Invalid syntax: set_local_or_remote_url(${name} ${ARGN})")
    endif()
    if(arg_LOCAL_URL AND (EXISTS ${arg_LOCAL_URL}))
        message(STATUS "Using local url: ${arg_LOCAL_URL}")
        set(${URL} "${arg_LOCAL_URL}" PARENT_SCOPE)
    else()
        message(STATUS "Using remote url(s): ${arg_REMOTE_URLS}")
        set(${URL} "${arg_REMOTE_URLS}" PARENT_SCOPE)
    endif()
endfunction()

include(ProcessorCount)
ProcessorCount(NPROC)

# Threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE) # -pthread instead of -lpthread
find_package(Threads REQUIRED)
list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "Threads")

# Assimp
message(STATUS "Building library Assimp from source")
include(${Open3D_3RDPARTY_DIR}/assimp/assimp.cmake)
import_3rdparty_library(3rdparty_assimp
    INCLUDE_DIRS ${ASSIMP_INCLUDE_DIR}
    LIB_DIR      ${ASSIMP_LIB_DIR}
    LIBRARIES    ${ASSIMP_LIBRARIES}
)
set(ASSIMP_TARGET "3rdparty_assimp")
add_dependencies(3rdparty_assimp ext_assimp)
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${ASSIMP_TARGET}")

# OpenMP
if(WITH_OPENMP)
    find_package(OpenMP)
    if(TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Building with OpenMP")
        set(OPENMP_TARGET "OpenMP::OpenMP_CXX")
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${OPENMP_TARGET}")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "OpenMP")
        endif()
    endif()
endif()

# X11
if(UNIX AND NOT APPLE)
    find_package(X11 QUIET)
    if(X11_FOUND)
        add_library(3rdparty_x11 INTERFACE)
        target_link_libraries(3rdparty_x11 INTERFACE ${X11_X11_LIB} ${CMAKE_THREAD_LIBS_INIT})
        if(NOT BUILD_SHARED_LIBS)
            install(TARGETS 3rdparty_x11 EXPORT ${PROJECT_NAME}Targets)
        endif()
        set(X11_TARGET "3rdparty_x11")
    endif()
endif()

# Dirent
if(WIN32)
    message(STATUS "Building library 3rdparty_dirent from source (WIN32)")
    build_3rdparty_library(3rdparty_dirent DIRECTORY dirent)
    set(DIRENT_TARGET "3rdparty_dirent")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${DIRENT_TARGET}")
endif()

# Eigen3
if(USE_SYSTEM_EIGEN3)
    find_package(Eigen3)
    if(TARGET Eigen3::Eigen)
        message(STATUS "Using installed third-party library Eigen3 ${EIGEN3_VERSION_STRING}")
        # Eigen3 is a publicly visible dependency, so add it to the list of
        # modules we need to find in the Open3D config script.
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "Eigen3")
        set(EIGEN3_TARGET "Eigen3::Eigen")
    else()
        message(STATUS "Unable to find installed third-party library Eigen3")
        set(USE_SYSTEM_EIGEN3 OFF)
    endif()
endif()
if(NOT USE_SYSTEM_EIGEN3)
    build_3rdparty_library(3rdparty_eigen3 PUBLIC DIRECTORY Eigen INCLUDE_DIRS Eigen INCLUDE_ALL)
    set(EIGEN3_TARGET "3rdparty_eigen3")
endif()
list(APPEND Open3D_3RDPARTY_PUBLIC_TARGETS "${EIGEN3_TARGET}")

# Flann
if(USE_SYSTEM_FLANN)
    pkg_config_3rdparty_library(3rdparty_flann flann)
endif()
if(NOT USE_SYSTEM_FLANN OR NOT 3rdparty_flann_FOUND)
    build_3rdparty_library(3rdparty_flann DIRECTORY flann)
endif()
set(FLANN_TARGET "3rdparty_flann")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${FLANN_TARGET}")

# Nanoflann
build_3rdparty_library(3rdparty_nanoflann DIRECTORY nanoflann INCLUDE_DIRS include/ INCLUDE_ALL)
set(NANOFLANN_TARGET "3rdparty_nanoflann")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${NANOFLANN_TARGET}")

# GLEW
if(USE_SYSTEM_GLEW)
    find_package(GLEW)
    if(TARGET GLEW::GLEW)
        message(STATUS "Using installed third-party library GLEW ${GLEW_VERSION}")
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "GLEW")
        set(GLEW_TARGET "GLEW::GLEW")
    else()
        pkg_config_3rdparty_library(3rdparty_glew glew)
        if(3rdparty_glew_FOUND)
            set(GLEW_TARGET "3rdparty_glew")
        else()
            set(USE_SYSTEM_GLEW OFF)
        endif()
    endif()
endif()
if(NOT USE_SYSTEM_GLEW)
    build_3rdparty_library(3rdparty_glew HEADER DIRECTORY glew SOURCES src/glew.c INCLUDE_DIRS include/)
    if(ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(3rdparty_glew PUBLIC GLEW_OSMESA)
    endif()
    if(WIN32)
        target_compile_definitions(3rdparty_glew PUBLIC GLEW_STATIC)
    endif()
    set(GLEW_TARGET "3rdparty_glew")
endif()
list(APPEND Open3D_3RDPARTY_HEADER_TARGETS "${GLEW_TARGET}")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${GLEW_TARGET}")

# GLFW
if(USE_SYSTEM_GLFW)
    find_package(glfw3)
    if(TARGET glfw)
        message(STATUS "Using installed third-party library glfw3")
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "glfw3")
        set(GLFW_TARGET "glfw")
    else()
        pkg_config_3rdparty_library(3rdparty_glfw3 glfw3)
        if(3rdparty_glfw3_FOUND)
            set(GLFW_TARGET "3rdparty_glfw3")
        else()
            set(USE_SYSTEM_GLFW OFF)
        endif()
    endif()
endif()
if(NOT USE_SYSTEM_GLFW)
    message(STATUS "Building library 3rdparty_glfw3 from source")
    add_subdirectory(${Open3D_3RDPARTY_DIR}/GLFW)
    import_3rdparty_library(3rdparty_glfw3 HEADER INCLUDE_DIRS ${Open3D_3RDPARTY_DIR}/GLFW/include/ LIBRARIES glfw3)
    add_dependencies(3rdparty_glfw3 glfw)
    target_link_libraries(3rdparty_glfw3 INTERFACE Threads::Threads)
    if(UNIX AND NOT APPLE)
        if(X11_TARGET)
            target_link_libraries(3rdparty_glfw3 INTERFACE ${X11_TARGET})
        endif()
        find_library(RT_LIBRARY rt)
        if(RT_LIBRARY)
            target_link_libraries(3rdparty_glfw3 INTERFACE ${RT_LIBRARY})
        endif()
        find_library(MATH_LIBRARY m)
        if(MATH_LIBRARY)
            target_link_libraries(3rdparty_glfw3 INTERFACE ${MATH_LIBRARY})
        endif()
        if(CMAKE_DL_LIBS)
            target_link_libraries(3rdparty_glfw3 INTERFACE ${CMAKE_DL_LIBS})
        endif()
    endif()
    if(APPLE)
        find_library(COCOA_FRAMEWORK Cocoa)
        find_library(IOKIT_FRAMEWORK IOKit)
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
        find_library(CORE_VIDEO_FRAMEWORK CoreVideo)
        target_link_libraries(3rdparty_glfw3 INTERFACE ${COCOA_FRAMEWORK} ${IOKIT_FRAMEWORK} ${CORE_FOUNDATION_FRAMEWORK} ${CORE_VIDEO_FRAMEWORK})
    endif()
    if(WIN32)
        target_link_libraries(3rdparty_glfw3 INTERFACE gdi32)
    endif()
    set(GLFW_TARGET "3rdparty_glfw3")
endif()
list(APPEND Open3D_3RDPARTY_HEADER_TARGETS "${GLFW_TARGET}")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${GLFW_TARGET}")

# TurboJPEG
if(USE_SYSTEM_JPEG AND BUILD_AZURE_KINECT)
    pkg_config_3rdparty_library(3rdparty_turbojpeg turbojpeg)
    if(3rdparty_turbojpeg_FOUND)
        message(STATUS "Using installed third-party library turbojpeg")
        set(TURBOJPEG_TARGET "3rdparty_turbojpeg")
    else()
        message(STATUS "Unable to find installed third-party library turbojpeg")
        message(STATUS "Azure Kinect driver needs TurboJPEG API")
        set(USE_SYSTEM_JPEG OFF)
    endif()
endif()

# JPEG
if(USE_SYSTEM_JPEG)
    find_package(JPEG)
    if(TARGET JPEG::JPEG)
        message(STATUS "Using installed third-party library JPEG")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "JPEG")
        endif()
        set(JPEG_TARGET "JPEG::JPEG")
        if(TURBOJPEG_TARGET)
            list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TURBOJPEG_TARGET}")
        endif()
    else()
        message(STATUS "Unable to find installed third-party library JPEG")
        set(USE_SYSTEM_JPEG OFF)
    endif()
endif()
if(NOT USE_SYSTEM_JPEG)
    message(STATUS "Building third-party library JPEG from source")
    include(${Open3D_3RDPARTY_DIR}/libjpeg-turbo/libjpeg-turbo.cmake)
    import_3rdparty_library(3rdparty_jpeg
        INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-install/include/
        LIBRARIES ${JPEG_TURBO_LIBRARIES}
        LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-install/lib
    )
    add_dependencies(3rdparty_jpeg ext_turbojpeg)
    set(JPEG_TARGET "3rdparty_jpeg")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${JPEG_TARGET}")

# jsoncpp: always compile from source to avoid ABI issues.
include(${Open3D_3RDPARTY_DIR}/jsoncpp/jsoncpp.cmake)
import_3rdparty_library(3rdparty_jsoncpp
    INCLUDE_DIRS ${JSONCPP_INCLUDE_DIRS}
    LIB_DIR      ${JSONCPP_LIB_DIR}
    LIBRARIES    ${JSONCPP_LIBRARIES}
)
set(JSONCPP_TARGET "3rdparty_jsoncpp")
add_dependencies(3rdparty_jsoncpp ext_jsoncpp)
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${JSONCPP_TARGET}")

# liblzf
if(USE_SYSTEM_LIBLZF)
    find_package(liblzf)
    if(TARGET liblzf::liblzf)
        message(STATUS "Using installed third-party library liblzf")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "liblzf")
        endif()
        set(LIBLZF_TARGET "liblzf::liblzf")
    else()
        message(STATUS "Unable to find installed third-party library liblzf")
        set(USE_SYSTEM_LIBLZF OFF)
    endif()
endif()
if(NOT USE_SYSTEM_LIBLZF)
    build_3rdparty_library(3rdparty_lzf DIRECTORY liblzf
        SOURCES
            liblzf/lzf_c.c
            liblzf/lzf_d.c
    )
    set(LIBLZF_TARGET "3rdparty_lzf")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${LIBLZF_TARGET}")

# tritriintersect
build_3rdparty_library(3rdparty_tritriintersect DIRECTORY tomasakeninemoeller INCLUDE_DIRS include/)
set(TRITRIINTERSECT_TARGET "3rdparty_tritriintersect")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TRITRIINTERSECT_TARGET}")

# librealsense SDK
if (BUILD_LIBREALSENSE)
    include(${Open3D_3RDPARTY_DIR}/librealsense/librealsense.cmake)
    import_3rdparty_library(3rdparty_librealsense
        INCLUDE_DIRS ${LIBREALSENSE_INCLUDE_DIR}
        LIBRARIES    ${LIBREALSENSE_LIBRARIES}
        LIB_DIR      ${LIBREALSENSE_LIB_DIR}
    )
    add_dependencies(3rdparty_librealsense ext_librealsense)
    set(LIBREALSENSE_TARGET "3rdparty_librealsense")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${LIBREALSENSE_TARGET}")

    if (UNIX AND NOT APPLE)    # Ubuntu dependency: libudev-dev
        find_library(UDEV_LIBRARY udev REQUIRED
            DOC "Library provided by the deb package libudev-dev")
        target_link_libraries(3rdparty_librealsense INTERFACE ${UDEV_LIBRARY})
    endif()
endif()

# PNG
if(USE_SYSTEM_PNG)
    find_package(PNG)
    if(TARGET PNG::PNG)
        message(STATUS "Using installed third-party library libpng")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "PNG")
        endif()
        set(PNG_TARGET "PNG::PNG")
    else()
        message(STATUS "Unable to find installed third-party library libpng")
        set(USE_SYSTEM_PNG OFF)
    endif()
endif()
if(NOT USE_SYSTEM_PNG)
    include(${Open3D_3RDPARTY_DIR}/zlib/zlib.cmake)
    import_3rdparty_library(3rdparty_zlib
        INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS}
        LIB_DIR      ${ZLIB_LIB_DIR}
        LIBRARIES    ${ZLIB_LIBRARIES}
    )
    set(ZLIB_TARGET "3rdparty_zlib")
    add_dependencies(3rdparty_zlib ext_zlib)

    include(${Open3D_3RDPARTY_DIR}/libpng/libpng.cmake)
    import_3rdparty_library(3rdparty_libpng
        INCLUDE_DIRS ${LIBPNG_INCLUDE_DIRS}
        LIB_DIR      ${LIBPNG_LIB_DIR}
        LIBRARIES    ${LIBPNG_LIBRARIES}
    )
    set(LIBPNG_TARGET "3rdparty_libpng")
    add_dependencies(3rdparty_libpng ext_libpng)
    add_dependencies(ext_libpng ext_zlib)

    # Putting zlib after libpng somehow works for Ubuntu.
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${LIBPNG_TARGET}")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${ZLIB_TARGET}")
endif()

# rply
build_3rdparty_library(3rdparty_rply DIRECTORY rply SOURCES rply/rply.c INCLUDE_DIRS rply/)
set(RPLY_TARGET "3rdparty_rply")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${RPLY_TARGET}")

# tinyfiledialogs
build_3rdparty_library(3rdparty_tinyfiledialogs
    DIRECTORY tinyfiledialogs
    SOURCES include/tinyfiledialogs/tinyfiledialogs.c
    INCLUDE_DIRS include/
)
set(TINYFILEDIALOGS_TARGET "3rdparty_tinyfiledialogs")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TINYFILEDIALOGS_TARGET}")

# tinygltf
if(USE_SYSTEM_TINYGLTF)
    find_package(TinyGLTF)
    if(TARGET TinyGLTF::TinyGLTF)
        message(STATUS "Using installed third-party library TinyGLTF")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "TinyGLTF")
        endif()
        set(TINYGLTF_TARGET "TinyGLTF::TinyGLTF")
    else()
        message(STATUS "Unable to find installed third-party library TinyGLTF")
        set(USE_SYSTEM_TINYGLTF OFF)
    endif()
endif()
if(NOT USE_SYSTEM_TINYGLTF)
    build_3rdparty_library(3rdparty_tinygltf DIRECTORY tinygltf INCLUDE_DIRS tinygltf/)
    target_compile_definitions(3rdparty_tinygltf INTERFACE TINYGLTF_IMPLEMENTATION STB_IMAGE_IMPLEMENTATION STB_IMAGE_WRITE_IMPLEMENTATION)
    set(TINYGLTF_TARGET "3rdparty_tinygltf")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TINYGLTF_TARGET}")

# tinyobjloader
if(USE_SYSTEM_TINYOBJLOADER)
    find_package(tinyobjloader)
    if(TARGET tinyobjloader::tinyobjloader)
        message(STATUS "Using installed third-party library tinyobjloader")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "tinyobjloader")
        endif()
        set(TINYOBJLOADER_TARGET "tinyobjloader::tinyobjloader")
    else()
        message(STATUS "Unable to find installed third-party library tinyobjloader")
        set(USE_SYSTEM_TINYOBJLOADER OFF)
    endif()
endif()
if(NOT USE_SYSTEM_TINYOBJLOADER)
    build_3rdparty_library(3rdparty_tinyobjloader DIRECTORY tinyobjloader INCLUDE_DIRS tinyobjloader/)
    target_compile_definitions(3rdparty_tinyobjloader INTERFACE TINYOBJLOADER_IMPLEMENTATION)
    set(TINYOBJLOADER_TARGET "3rdparty_tinyobjloader")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TINYOBJLOADER_TARGET}")

# Qhull
if(USE_SYSTEM_QHULL)
    find_package(Qhull)
    if(TARGET Qhull::qhullcpp)
        message(STATUS "Using installed third-party library Qhull")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "Qhull")
        endif()
        set(QHULL_TARGET "Qhull::qhullcpp")
    else()
        message(STATUS "Unable to find installed third-party library Qhull")
        set(USE_SYSTEM_QHULL OFF)
    endif()
endif()
if(NOT USE_SYSTEM_QHULL)
    build_3rdparty_library(3rdparty_qhull_r DIRECTORY qhull
        SOURCES
            src/libqhull_r/global_r.c
            src/libqhull_r/stat_r.c
            src/libqhull_r/geom2_r.c
            src/libqhull_r/poly2_r.c
            src/libqhull_r/merge_r.c
            src/libqhull_r/libqhull_r.c
            src/libqhull_r/geom_r.c
            src/libqhull_r/poly_r.c
            src/libqhull_r/qset_r.c
            src/libqhull_r/mem_r.c
            src/libqhull_r/random_r.c
            src/libqhull_r/usermem_r.c
            src/libqhull_r/userprintf_r.c
            src/libqhull_r/io_r.c
            src/libqhull_r/user_r.c
            src/libqhull_r/rboxlib_r.c
            src/libqhull_r/userprintf_rbox_r.c
        INCLUDE_DIRS
            src/
    )
    build_3rdparty_library(3rdparty_qhullcpp DIRECTORY qhull
        SOURCES
            src/libqhullcpp/Coordinates.cpp
            src/libqhullcpp/PointCoordinates.cpp
            src/libqhullcpp/Qhull.cpp
            src/libqhullcpp/QhullFacet.cpp
            src/libqhullcpp/QhullFacetList.cpp
            src/libqhullcpp/QhullFacetSet.cpp
            src/libqhullcpp/QhullHyperplane.cpp
            src/libqhullcpp/QhullPoint.cpp
            src/libqhullcpp/QhullPointSet.cpp
            src/libqhullcpp/QhullPoints.cpp
            src/libqhullcpp/QhullQh.cpp
            src/libqhullcpp/QhullRidge.cpp
            src/libqhullcpp/QhullSet.cpp
            src/libqhullcpp/QhullStat.cpp
            src/libqhullcpp/QhullVertex.cpp
            src/libqhullcpp/QhullVertexSet.cpp
            src/libqhullcpp/RboxPoints.cpp
            src/libqhullcpp/RoadError.cpp
            src/libqhullcpp/RoadLogEvent.cpp
        INCLUDE_DIRS
            src/
    )
    target_link_libraries(3rdparty_qhullcpp PRIVATE 3rdparty_qhull_r)
    set(QHULL_TARGET "3rdparty_qhullcpp")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${QHULL_TARGET}")

# fmt
if(USE_SYSTEM_FMT)
    find_package(fmt)
    if(TARGET fmt::fmt-header-only)
        message(STATUS "Using installed third-party library fmt (header only)")
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "fmt")
        set(FMT_TARGET "fmt::fmt-header-only")
    elseif(TARGET fmt::fmt)
        message(STATUS "Using installed third-party library fmt")
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "fmt")
        set(FMT_TARGET "fmt::fmt")
    else()
        message(STATUS "Unable to find installed third-party library fmt")
        set(USE_SYSTEM_FMT OFF)
    endif()
endif()
if(NOT USE_SYSTEM_FMT)
    # We set the FMT_HEADER_ONLY macro, so no need to actually compile the source
    build_3rdparty_library(3rdparty_fmt PUBLIC DIRECTORY fmt INCLUDE_DIRS include/)
    target_compile_definitions(3rdparty_fmt INTERFACE FMT_HEADER_ONLY=1)
    set(FMT_TARGET "3rdparty_fmt")
endif()
list(APPEND Open3D_3RDPARTY_PUBLIC_TARGETS "${FMT_TARGET}")

# Pybind11
if(USE_SYSTEM_PYBIND11)
    find_package(pybind11)
endif()
if (NOT USE_SYSTEM_PYBIND11 OR NOT TARGET pybind11::module)
    set(USE_SYSTEM_PYBIND11 OFF)
    add_subdirectory(${Open3D_3RDPARTY_DIR}/pybind11)
endif()
if(TARGET pybind11::module)
    set(PYBIND11_TARGET "pybind11::module")
endif()

# Azure Kinect
set(BUILD_AZURE_KINECT_COMMENT "//") # Set include header files in Open3D.h
if (BUILD_AZURE_KINECT)
    include(${Open3D_3RDPARTY_DIR}/azure_kinect/azure_kinect.cmake)
    import_3rdparty_library(3rdparty_k4a
        INCLUDE_DIRS ${K4A_INCLUDE_DIR}
    )
    add_dependencies(3rdparty_k4a ext_k4a)
    set(K4A_TARGET "3rdparty_k4a")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${K4A_TARGET}")
endif()

# PoissonRecon
build_3rdparty_library(3rdparty_poisson DIRECTORY PoissonRecon INCLUDE_DIRS PoissonRecon)
set(POISSON_TARGET "3rdparty_poisson")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${POISSON_TARGET}")

# Googletest
if (BUILD_UNIT_TESTS)
    if(USE_SYSTEM_GOOGLETEST)
        find_path(gtest_INCLUDE_DIRS gtest/gtest.h)
        find_library(gtest_LIBRARY gtest)
        find_path(gmock_INCLUDE_DIRS gmock/gmock.h)
        find_library(gmock_LIBRARY gmock)
        if(gtest_INCLUDE_DIRS AND gtest_LIBRARY AND gmock_INCLUDE_DIRS AND gmock_LIBRARY)
            message(STATUS "Using installed googletest")
            add_library(3rdparty_googletest INTERFACE)
            target_include_directories(3rdparty_googletest INTERFACE ${gtest_INCLUDE_DIRS} ${gmock_INCLUDE_DIRS})
            target_link_libraries(3rdparty_googletest INTERFACE ${gtest_LIBRARY} ${gmock_LIBRARY})
            set(GOOGLETEST_TARGET "3rdparty_googletest")
        else()
            message(STATUS "Unable to find installed googletest")
            set(USE_SYSTEM_GOOGLETEST OFF)
        endif()
    endif()
    if(NOT USE_SYSTEM_GOOGLETEST)
        build_3rdparty_library(3rdparty_googletest DIRECTORY googletest
            SOURCES
                googletest/src/gtest-all.cc
                googlemock/src/gmock-all.cc
            INCLUDE_DIRS
                googletest/include/
                googletest/
                googlemock/include/
                googlemock/
        )
        set(GOOGLETEST_TARGET "3rdparty_googletest")
    endif()
endif()

# Headless rendering
if (ENABLE_HEADLESS_RENDERING)
    find_package(OSMesa REQUIRED)
    add_library(3rdparty_osmesa INTERFACE)
    target_include_directories(3rdparty_osmesa INTERFACE ${OSMESA_INCLUDE_DIR})
    target_link_libraries(3rdparty_osmesa INTERFACE ${OSMESA_LIBRARY})
    if(NOT BUILD_SHARED_LIBS)
        install(TARGETS 3rdparty_osmesa EXPORT ${PROJECT_NAME}Targets
        RUNTIME DESTINATION ${Open3D_INSTALL_BIN_DIR}
        ARCHIVE DESTINATION ${Open3D_INSTALL_LIB_DIR}
        LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
    )
    endif()
    set(OPENGL_TARGET "3rdparty_osmesa")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${OPENGL_TARGET}")
else()
    find_package(OpenGL)
    if(TARGET OpenGL::GL)
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "OpenGL")
        endif()
        set(OPENGL_TARGET "OpenGL::GL")
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${OPENGL_TARGET}")
    endif()
endif()

# imgui
if(BUILD_GUI)
    if(USE_SYSTEM_IMGUI)
        find_package(ImGui)
        if(TARGET ImGui::ImGui)
            message(STATUS "Using installed third-party library ImGui")
            if(NOT BUILD_SHARED_LIBS)
                list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "ImGui")
            endif()
            set(IMGUI_TARGET "ImGui::ImGui")
        else()
            message(STATUS "Unable to find installed third-party library ImGui")
            set(USE_SYSTEM_IMGUI OFF)
        endif()
    endif()
    if(NOT USE_SYSTEM_IMGUI)
        build_3rdparty_library(3rdparty_imgui DIRECTORY imgui
            SOURCES
                imgui_demo.cpp
                imgui_draw.cpp
                imgui_widgets.cpp
                imgui.cpp
        )
        set(IMGUI_TARGET "3rdparty_imgui")
    endif()
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${IMGUI_TARGET}")
endif()

# Filament
if(BUILD_GUI)
    if(BUILD_FILAMENT_FROM_SOURCE)
        message(STATUS "Building third-party library Filament from source")
        if(MSVC OR (CMAKE_C_COMPILER_ID MATCHES ".*Clang" AND
            CMAKE_CXX_COMPILER_ID MATCHES ".*Clang"
            AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7))
            set(FILAMENT_C_COMPILER "${CMAKE_C_COMPILER}")
            set(FILAMENT_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
        else()
            message(STATUS "Filament can only be built with Clang >= 7")
            # First, check default version, because the user may have configured
            # a particular version as default for a reason.
            find_program(CLANG_DEFAULT_CC NAMES clang)
            find_program(CLANG_DEFAULT_CXX NAMES clang++)
            if(CLANG_DEFAULT_CC AND CLANG_DEFAULT_CXX)
                execute_process(COMMAND ${CLANG_DEFAULT_CXX} --version OUTPUT_VARIABLE clang_version)
                if(clang_version MATCHES "clang version ([0-9]+)")
                    if (CMAKE_MATCH_1 GREATER_EQUAL 7)
                        message(STATUS "Using ${CLANG_DEFAULT_CXX} to build Filament")
                        set(FILAMENT_C_COMPILER "${CLANG_DEFAULT_CC}")
                        set(FILAMENT_CXX_COMPILER "${CLANG_DEFAULT_CXX}")
                    endif()
                endif()
            endif()
            # If the default version is not sufficient, look for some specific versions
            if(NOT FILAMENT_C_COMPILER OR NOT FILAMENT_CXX_COMPILER)
                find_program(CLANG_VERSIONED_CC NAMES clang-11 clang-10 clang-9 clang-8 clang-7)
                find_program(CLANG_VERSIONED_CXX NAMES clang++11 clang++-10 clang++-9 clang++-8 clang++-7)
                if (CLANG_VERSIONED_CC AND CLANG_VERSIONED_CXX)
                    set(FILAMENT_C_COMPILER "${CLANG_VERSIONED_CC}")
                    set(FILAMENT_CXX_COMPILER "${CLANG_VERSIONED_CXX}")
                    message(STATUS "Using ${CLANG_VERSIONED_CXX} to build Filament")
                else()
                    message(FATAL_ERROR "Need Clang >= 7 to compile Filament from source")
                endif()
            endif()
        endif()
        # Find corresponding libc++ and libc++abi libraries. On Ubuntu, clang
        # libraries are located at /usr/lib/llvm-{version}/lib, and the default
        # version will have a sybolic link at /usr/lib/x86_64-linux-gnu/ or
        # /usr/lib/aarch64-linux-gnu.
        # For aarch64, the symbolic link path may not work for CMake's
        # find_library. Therefore, when compiling Filament from source, we
        # explicitly find the corresponidng path based on the clang version.
        execute_process(COMMAND ${FILAMENT_CXX_COMPILER} --version OUTPUT_VARIABLE clang_version)
        if(clang_version MATCHES "clang version ([0-9]+)")
            set(CLANG_LIBDIR "/usr/lib/llvm-${CMAKE_MATCH_1}/lib")
        endif()
        include(${Open3D_3RDPARTY_DIR}/filament/filament_build.cmake)
    else()
        message(STATUS "Using prebuilt third-party library Filament")
        include(${Open3D_3RDPARTY_DIR}/filament/filament_download.cmake)
    endif()
    set(FILAMENT_RUNTIME_VER "")
    if (WIN32)
        if (STATIC_WINDOWS_RUNTIME)
            set(FILAMENT_RUNTIME_VER "mt$<$<CONFIG:DEBUG>:d>")
        else()
            set(FILAMENT_RUNTIME_VER "md$<$<CONFIG:DEBUG>:d>")
        endif()
    endif()
    import_3rdparty_library(3rdparty_filament HEADER
        INCLUDE_DIRS ${FILAMENT_ROOT}/include/
        LIB_DIR ${FILAMENT_ROOT}/lib/x86_64/${FILAMENT_RUNTIME_VER}
        LIBRARIES ${filament_LIBRARIES}
    )
    set(FILAMENT_MATC "${FILAMENT_ROOT}/bin/matc")
    target_link_libraries(3rdparty_filament INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
    if(UNIX AND NOT APPLE)
        # Find CLANG_LIBDIR if it is not defined. Mutiple paths will be searched.
        if (NOT CLANG_LIBDIR)
            find_library(CPPABI_LIBRARY c++abi PATH_SUFFIXES
                         llvm-11/lib llvm-10/lib llvm-9/lib llvm-8/lib llvm-7/lib
                         REQUIRED)
            get_filename_component(CLANG_LIBDIR ${CPPABI_LIBRARY} DIRECTORY)
        endif()
        # Find clang libraries at the exact path ${CLANG_LIBDIR}.
        find_library(CPP_LIBRARY    c++    PATHS ${CLANG_LIBDIR} REQUIRED NO_DEFAULT_PATH)
        find_library(CPPABI_LIBRARY c++abi PATHS ${CLANG_LIBDIR} REQUIRED NO_DEFAULT_PATH)
        if(CPP_LIBRARY-NOTFOUND)
            message(FATAL_ERROR "CPP_LIBRARY-NOTFOUND")
        endif()
        if(CPPABI_LIBRARY-NOTFOUND)
            message(FATAL_ERROR "CPPABI_LIBRARY-NOTFOUND")
        endif()
        # Ensure that libstdc++ gets linked first
        target_link_libraries(3rdparty_filament INTERFACE -lstdc++
                              ${CPP_LIBRARY} ${CPPABI_LIBRARY})
        message(STATUS "CLANG_LIBDIR: ${CLANG_LIBDIR}")
        message(STATUS "CPP_LIBRARY: ${CPP_LIBRARY}")
        message(STATUS "CPPABI_LIBRARY: ${CPPABI_LIBRARY}")
    endif()
    if (APPLE)
        find_library(CORE_VIDEO CoreVideo)
        find_library(QUARTZ_CORE QuartzCore)
        find_library(OPENGL_LIBRARY OpenGL)
        find_library(METAL_LIBRARY Metal)
        find_library(APPKIT_LIBRARY AppKit)
        target_link_libraries(3rdparty_filament INTERFACE ${CORE_VIDEO} ${QUARTZ_CORE} ${OPENGL_LIBRARY} ${METAL_LIBRARY} ${APPKIT_LIBRARY})
        target_link_options(3rdparty_filament INTERFACE "-fobjc-link-runtime")
    endif()
    if(TARGET ext_filament)
        # Make sure that the external project is built first
        add_dependencies(3rdparty_filament ext_filament)
    endif()
    set(FILAMENT_TARGET "3rdparty_filament")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${FILAMENT_TARGET}")
endif()

# RPC interface
# - boost: predef
# - zeromq
# - msgpack
if(BUILD_RPC_INTERFACE)
    # boost: predef
    include(${Open3D_3RDPARTY_DIR}/boost/boost.cmake)
    import_3rdparty_library(3rdparty_boost
        INCLUDE_DIRS ${BOOST_INCLUDE_DIRS}
    )
    set(BOOST_TARGET "3rdparty_boost")
    add_dependencies(3rdparty_boost ext_boost)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${BOOST_TARGET}")

    # zeromq
    include(${Open3D_3RDPARTY_DIR}/zeromq/zeromq_build.cmake)
    import_3rdparty_library(3rdparty_zeromq
        INCLUDE_DIRS ${ZEROMQ_INCLUDE_DIRS}
        LIB_DIR ${ZEROMQ_LIB_DIR}
        LIBRARIES ${ZEROMQ_LIBRARIES}
    )
    set(ZEROMQ_TARGET "3rdparty_zeromq")
    add_dependencies(${ZEROMQ_TARGET} ext_zeromq)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${ZEROMQ_TARGET}")
    if( DEFINED ZEROMQ_ADDITIONAL_LIBS )
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS ${ZEROMQ_ADDITIONAL_LIBS})
    endif()

    # msgpack
    include(${Open3D_3RDPARTY_DIR}/msgpack/msgpack_build.cmake)
    import_3rdparty_library(3rdparty_msgpack
        INCLUDE_DIRS ${MSGPACK_INCLUDE_DIRS}
    )
    set(MSGPACK_TARGET "3rdparty_msgpack")
    add_dependencies(3rdparty_msgpack ext_msgpack-c)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${MSGPACK_TARGET}")
endif()

# TBB
include(${Open3D_3RDPARTY_DIR}/mkl/tbb.cmake)
import_3rdparty_library(3rdparty_tbb
    INCLUDE_DIRS ${STATIC_TBB_INCLUDE_DIR}
    LIB_DIR      ${STATIC_TBB_LIB_DIR}
    LIBRARIES    ${STATIC_TBB_LIBRARIES}
)
set(TBB_TARGET "3rdparty_tbb")
add_dependencies(3rdparty_tbb ext_tbb)
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TBB_TARGET}")

if(USE_BLAS)
    # Try to locate system BLAS/LAPACK
    find_package(BLAS)
    find_package(LAPACK)
    find_package(LAPACKE)
    if(BLAS_FOUND AND LAPACK_FOUND AND LAPACKE_FOUND)
        message(STATUS "Using system BLAS/LAPACK")
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${BLAS_LIBRARIES}")
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${LAPACK_LIBRARIES}")
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${LAPACKE_LIBRARIES}")
        if(BUILD_CUDA_MODULE)
            list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${CUDA_cusolver_LIBRARY}")
            list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${CUDA_CUBLAS_LIBRARIES}")
        endif()
    else()
        # Compile OpenBLAS/Lapack from source. Install gfortran on Ubuntu first.
        message(STATUS "Building OpenBLAS with LAPACK from source")
        set(BLAS_BUILD_FROM_SOURCE ON)

        include(${Open3D_3RDPARTY_DIR}/openblas/openblas.cmake)
        import_3rdparty_library(3rdparty_openblas
            INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR}
            LIB_DIR      ${OPENBLAS_LIB_DIR}
            LIBRARIES    ${OPENBLAS_LIBRARIES}
        )
        set(OPENBLAS_TARGET "3rdparty_openblas")
        add_dependencies(3rdparty_openblas ext_openblas)
        target_link_libraries(3rdparty_openblas INTERFACE Threads::Threads gfortran)
        if(BUILD_CUDA_MODULE)
            target_link_libraries(3rdparty_openblas INTERFACE
                                ${CUDA_cusolver_LIBRARY}
                                ${CUDA_CUBLAS_LIBRARIES})
        endif()
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${OPENBLAS_TARGET}")
    endif()
else()
    include(${Open3D_3RDPARTY_DIR}/mkl/mkl.cmake)
    # MKL, cuSOLVER, cuBLAS
    # We link MKL statically. For MKL link flags, refer to:
    # https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html
    message(STATUS "Using MKL to support BLAS and LAPACK functionalities.")
    import_3rdparty_library(3rdparty_mkl
        INCLUDE_DIRS ${STATIC_MKL_INCLUDE_DIR}
        LIB_DIR      ${STATIC_MKL_LIB_DIR}
        LIBRARIES    ${STATIC_MKL_LIBRARIES}
    )
    set(MKL_TARGET "3rdparty_mkl")
    add_dependencies(3rdparty_mkl ext_tbb ext_mkl_include ext_mkl)

    message(STATUS "STATIC_MKL_INCLUDE_DIR: ${STATIC_MKL_INCLUDE_DIR}")
    message(STATUS "STATIC_MKL_LIB_DIR: ${STATIC_MKL_LIB_DIR}")
    message(STATUS "STATIC_MKL_LIBRARIES: ${STATIC_MKL_LIBRARIES}")
    if(UNIX)
        target_compile_options(3rdparty_mkl INTERFACE "-DMKL_ILP64 -m64")
        target_link_libraries(3rdparty_mkl INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
        # cuSOLVER and cuBLAS
        if(BUILD_CUDA_MODULE)
            target_link_libraries(3rdparty_mkl INTERFACE
                                ${CUDA_cusolver_LIBRARY}
                                ${CUDA_CUBLAS_LIBRARIES})
        endif()
    elseif(MSVC)
        target_compile_options(3rdparty_mkl INTERFACE "/DMKL_ILP64")
    endif()
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${MKL_TARGET}")
endif()

# Faiss
if (WITH_FAISS AND WIN32)
    message(STATUS "Faiss is not supported on Windows")
    set(WITH_FAISS OFF)
elseif(WITH_FAISS)
    message(STATUS "Building third-party library faiss from source")
    include(${Open3D_3RDPARTY_DIR}/faiss/faiss_build.cmake)
endif()
if (WITH_FAISS)
    message(STATUS "FAISS_INCLUDE_DIR: ${FAISS_INCLUDE_DIR}")
    message(STATUS "FAISS_LIB_DIR: ${FAISS_LIB_DIR}")
    if (USE_BLAS)
        if (BLAS_BUILD_FROM_SOURCE)
            set(FAISS_EXTRA_DEPENDENCIES 3rdparty_openblas)
        endif()
    else()
        set(FAISS_EXTRA_LIBRARIES ${STATIC_MKL_LIBRARIES})
        set(FAISS_EXTRA_DEPENDENCIES 3rdparty_mkl)
    endif()
    import_3rdparty_library(3rdparty_faiss
        INCLUDE_DIRS ${FAISS_INCLUDE_DIR}
        LIBRARIES ${FAISS_LIBRARIES} ${FAISS_EXTRA_LIBRARIES}
        LIB_DIR ${FAISS_LIB_DIR}
    )
    add_dependencies(3rdparty_faiss ext_faiss)
    if (FAISS_EXTRA_DEPENDENCIES)
        add_dependencies(ext_faiss ${FAISS_EXTRA_DEPENDENCIES})
    endif()
    set(FAISS_TARGET "3rdparty_faiss")
    target_link_libraries(3rdparty_faiss INTERFACE ${CMAKE_DL_LIBS})
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${FAISS_TARGET}")
