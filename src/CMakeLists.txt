# References:
# https://cmake.org/cmake/help/v3.0/
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/

cmake_minimum_required(VERSION 3.0.0)

if (NOT CMAKE_VERSION VERSION_LESS "3.0")
    cmake_policy(SET CMP0054 OLD)
endif ()

# central location for specifying the Open3D version
project(Open3D
        VERSION 0.1.5.0)

# set additional info
set(PROJECT_EMAIL   "info@open3d.org")
set(PROJECT_HOME    "http://www.open3d.org")
set(PROJECT_DOCS    "http://www.open3d.org/docs")
set(PROJECT_CODE    "https://github.com/IntelVCL/Open3D")
set(PROJECT_ISSUES  "https://github.com/IntelVCL/Open3D/issues")

# configure a header file to pass the version settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/open3d_config.h.in"
               "${PROJECT_SOURCE_DIR}/open3d_config.h")

# configure a setup.py for the pip packaging
configure_file("${PROJECT_SOURCE_DIR}/../util/pip_package/setup.py.in"
               "${PROJECT_SOURCE_DIR}/../util/pip_package/setup.py")

# don't set the CMAKE_INSTALL_PREFIX here
# let the user pick either the OS-specific default path or a custom path
# Windows test location: "C:\\Open3D"
# macOS test location:   "~/Library/"
# Linux test location:   "~/.local/"

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# config options
option(BUILD_SHARED_LIBS         "Build shared libraries"                   OFF)
option(WITH_OPENMP               "Use OpenMP multi-threading"               ON)
option(ENABLE_HEADLESS_RENDERING "Use OSMesa for headless rendering"        OFF)
option(BUILD_TESTS               "Build the Open3D test programs"           ON)
option(BUILD_UNIT_TESTS          "Build the Open3D unit tests"              OFF)
option(BUILD_EXPERIMENTAL        "Build the Open3D experimental programs"   ON)
option(BUILD_EIGEN3              "Build eigen3 from source"                 OFF)
option(BUILD_GLEW                "Build glew from source"                   OFF)
option(BUILD_GLFW                "Build glfw from source"                   OFF)
option(BUILD_JPEG                "Build jpeg from source"                   OFF)
option(BUILD_JSONCPP             "Build json from source"                   OFF)
option(BUILD_PNG                 "Build png from source"                    OFF)
option(BUILD_PYBIND11            "Build pybind11 from source"               ON)
option(BUILD_PYTHON_MODULE       "Build the python module"                  ON)
option(BUILD_PYTHON_TESTS        "Copy the python test files"               ON)
option(BUILD_PYTHON_TUTORIALS    "Copy the python tutorial files"           ON)
option(BUILD_LIBREALSENSE        "Build support for Intel RealSense camera" OFF)
option(BUILD_TINYFILEDIALOGS     "Build tinyfiledialogs from source"        ON)

# default built type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif ()

# set include directories
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/External
    ${PROJECT_SOURCE_DIR}/External/librealsense/include
    )

# if dynamic link is added, use if (WIN32) macro
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Set OS-specific things here
if (WIN32)
    # can't hide the unit testing option on Windows only
    # as a precaution: disable unit testing on Windows regardless of user input
    message(STATUS "Disable unit tests since this feature is not fully supported on Windows.")
    set(BUILD_UNIT_TESTS OFF)

    # build external dependencies from source
    set(BUILD_EIGEN3  ON)
    set(BUILD_GLEW    ON)
    set(BUILD_GLFW    ON)
    set(BUILD_JPEG    ON)
    set(BUILD_JSONCPP ON)
    set(BUILD_PNG     ON)

    add_definitions(-DWINDOWS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)		# suppress C4996 warning
    add_definitions(-DGLEW_STATIC)		# enable GLEW for windows
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Windows")
    if (MSVC)
        message(STATUS "Compiling with MSVC")
        add_definitions(-DNOMINMAX)
        add_definitions(-D_USE_MATH_DEFINES)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    endif (MSVC)
elseif (CYGWIN)
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Cygwin")
    # enable c++11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCYGWIN)
elseif (APPLE)
    add_definitions(-DUNIX)
    # enable c++11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    # compile speed optimization for clang
    add_definitions(-O3)
elseif (UNIX)
    add_definitions(-DUNIX)
    # enable c++11
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
    add_definitions(-O3)
    # disable BUILD_LIBREALSENSE since it is not fully supported on Linux
    message(STATUS "Compiling on Unix")
    message(STATUS "Disable RealSense since it is not fully supported on Linux.")
    set(BUILD_LIBREALSENSE OFF)
endif ()

# Set OpenMP
if (WITH_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        message(STATUS "Using installed OpenMP ${OpenMP_VERSION}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        message(STATUS "OpenMP NOT found")
    endif ()
endif ()

# Handling dependencies
add_subdirectory(External)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})
link_directories(${GLFW_LIBRARY_DIRS})
link_directories(${JSONCPP_LIBRARY_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

# Open3D libraries
add_subdirectory(Core)
add_subdirectory(Tools)
add_subdirectory(IO)
add_subdirectory(Visualization)

add_library(${PROJECT_NAME} $<TARGET_OBJECTS:Core> $<TARGET_OBJECTS:IO> $<TARGET_OBJECTS:Visualization>)

# headless rendering
if (ENABLE_HEADLESS_RENDERING)
    find_package(OSMesa REQUIRED)
    include_directories(${OSMESA_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${OSMESA_LIBRARY})
else ()
    find_package(OpenGL REQUIRED)
    target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES})
endif ()

target_link_libraries(${PROJECT_NAME}
                      ${GLFW_LIBRARIES}
                      ${GLEW_LIBRARIES}
                      ${JPEG_LIBRARIES}
                      ${JSONCPP_LIBRARIES}
                      ${OMP_LIBRARIES}
                      ${PNG_LIBRARIES}
                      tinyfiledialogs)

# install
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB HEADERS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS_ROOT} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})

# Tests
if (BUILD_TESTS)
    add_subdirectory(Test)
endif ()

# Unit tests
if (BUILD_UNIT_TESTS)
    add_subdirectory(UnitTest)
endif ()

# Experimental
if (BUILD_EXPERIMENTAL)
    message(STATUS "Build experimental projects")
    add_subdirectory(Experimental)
endif ()

# Python binding
if (BUILD_PYTHON_MODULE AND BUILD_PYBIND11)
    add_subdirectory(Python)
endif ()

# uninstall target
if (NOT TARGET uninstall)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
                   IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
                      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif ()
