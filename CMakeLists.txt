cmake_minimum_required(VERSION 3.8.0) # 3.8.0 to enable CUDA native support

# config options
option(BUILD_SHARED_LIBS         "Build shared libraries"                   OFF)
option(WITH_OPENMP               "Use OpenMP multi-threading"               ON)
option(ENABLE_HEADLESS_RENDERING "Use OSMesa for headless rendering"        OFF)
option(BUILD_CPP_EXAMPLES        "Build the Open3D example programs"        ON)
option(BUILD_UNIT_TESTS          "Build the Open3D unit tests"              OFF)
option(BUILD_BENCHMARKS          "Build the micro benchmarks"               OFF)
option(BUILD_EIGEN3              "Use the Eigen3 that comes with Open3D"    OFF)
option(BUILD_FLANN               "Use the flann that comes with Open3D"     ON)
option(BUILD_FMT                 "Build fmtlib from source"                 ON)
option(BUILD_GLEW                "Build glew from source"                   OFF)
option(BUILD_GLFW                "Build glfw from source"                   OFF)
option(BUILD_JSONCPP             "Build jsoncpp from source"                ON)
option(BUILD_LIBLZF              "Build liblzf from source"                 ON)
option(BUILD_PNG                 "Build png from source"                    OFF)
option(BUILD_ZLIB                "Build zlib from source"                   OFF)
option(BUILD_JPEG                "Build jpeg-turbo from source"             OFF)
option(BUILD_PYBIND11            "Build pybind11 from source"               ON)
option(BUILD_PYTHON_MODULE       "Build the python module"                  ON)
option(BUILD_LIBREALSENSE        "Build support for Intel RealSense camera" OFF)
option(BUILD_AZURE_KINECT        "Build support for Azure Kinect sensor"    OFF)
option(BUILD_TINYFILEDIALOGS     "Build tinyfiledialogs from source"        ON)
option(BUILD_TINYGLTF            "Build TinyGLTF from source"               ON)
option(BUILD_TINYOBJLOADER       "Build tinyobjloader from source"          ON)
option(BUILD_QHULL               "Build qhull from source"                  ON)
option(BUILD_CUDA_MODULE         "Build the CUDA module"                    OFF)
option(BUILD_TENSORFLOW_OPS      "Build ops for Tensorflow"                 OFF)
option(ENABLE_JUPYTER            "Enable Jupyter support for Open3D"        ON)
option(STATIC_WINDOWS_RUNTIME    "Use static (MT/MTd) Windows runtime"      OFF)
option(GLIBCXX_USE_CXX11_ABI     "Set -D_GLIBCXX_USE_CXX11_ABI=1"           OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No CMAKE_BUILD_TYPE specified, default to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# Override OpenGL policy
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
    if (BUILD_GLFW)
        cmake_policy(SET CMP0072 OLD)
    endif()
endif()

# Parse Open3D version number
file(STRINGS "src/Open3D/version.txt" OPEN3D_VERSION_READ)
foreach(ver ${OPEN3D_VERSION_READ})
    if (ver MATCHES "OPEN3D_VERSION_(MAJOR|MINOR|PATCH|TWEAK) +([^ ]+)$")
        set(OPEN3D_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
string(CONCAT OPEN3D_VERSION
    "${OPEN3D_VERSION_MAJOR}"
    ".${OPEN3D_VERSION_MINOR}"
    ".${OPEN3D_VERSION_PATCH}"
    ".${OPEN3D_VERSION_TWEAK}"
)

project(Open3D VERSION ${OPEN3D_VERSION} LANGUAGES C CXX)
message(STATUS "Open3D ${PROJECT_VERSION}")

# npm version has to be MAJOR.MINOR.PATCH
string(CONCAT PROJECT_VERSION_THREE_NUMBER "${OPEN3D_VERSION_MAJOR}"
                                           ".${OPEN3D_VERSION_MINOR}"
                                           ".${OPEN3D_VERSION_PATCH}")

# PyPI package name controls specifies the repository name on PyPI. The default
# name is "open3d". In the past, for historical reasons, we've used the
# following names for PyPI, while they are now deprecated:
# - open3d-python
# - py3d
# - open3d-original
# - open3d-official
# - open-3d
if(NOT DEFINED PYPI_PACKAGE_NAME)
    set(PYPI_PACKAGE_NAME "open3d")
endif()

# set additional info
set(PROJECT_EMAIL   "info@open3d.org")
set(PROJECT_HOME    "http://www.open3d.org")
set(PROJECT_DOCS    "http://www.open3d.org/docs")
set(PROJECT_CODE    "https://github.com/intel-isl/Open3D")
set(PROJECT_ISSUES  "https://github.com/intel-isl/Open3D/issues")

# Set installation paths
if(UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(Open3D_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(Open3D_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(Open3D_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(Open3D_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(Open3D_INSTALL_INCLUDE_DIR include)
    set(Open3D_INSTALL_BIN_DIR bin)
    set(Open3D_INSTALL_LIB_DIR lib)
    set(Open3D_INSTALL_CMAKE_DIR CMake)
endif()

# Put build results in some predictable places
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Build CUDA module by default if CUDA is available
# Compatible with CMake 3.8+
if (BUILD_CUDA_MODULE)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDA REQUIRED) # required for cuda_select_nvcc_arch_flags
        message(STATUS "Building CUDA enabled")
        enable_language(CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
        # get gencode flags
        if( "${CUDA_ARCH}" STREQUAL "User" )
            cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH_USER}")
        elseif( "${CUDA_ARCH}" STREQUAL "BasicPTX" )
            # include oldest and most recent PTX and rely on JIT compilation
            set(CUDA_GENCODES "-gencode arch=compute_30,code=compute_30;-gencode arch=compute_75,code=compute_75")
        else()
            cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH}")
        endif()
        string( REPLACE ";" " " CUDA_GENCODES "${CUDA_GENCODES}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODES}")
    else()
        set(BUILD_CUDA_MODULE OFF)
        message(STATUS "No CUDA support")
    endif()
endif ()

if(WIN32)
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explictly export symbols.
    #       Then, we could use -fvisibility=hidden for Linux as well
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Catch a few incompatible settings
if((UNIX AND NOT APPLE) AND BUILD_LIBREALSENSE)
    message(STATUS "Disabling RealSense since it is not fully supported on Linux.")
    set(BUILD_LIBREALSENSE OFF)
endif()

if (APPLE AND ENABLE_HEADLESS_RENDERING)
    message(STATUS "Headless rendering is not supported on Mac OS")
    set(ENABLE_HEADLESS_RENDERING OFF)
endif()

# CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/3rdparty/CMake)

# Folder view for project files
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Enumerate all third-party libraries which we need later
# This creates the nececssary targets and sets the 
# Open3D_3RDPARTY_*_TARGETS variables we use below
include(3rdparty/find_dependencies.cmake)

# Convenience function to link against all third-party libraries
# We need this because we create a lot of object libraries to assemble
# the main Open3D library
function(open3d_link_3rdparty_libraries target)
    target_link_libraries(${target} PRIVATE ${Open3D_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${Open3D_3RDPARTY_PUBLIC_TARGETS})
    foreach(dep IN LISTS Open3D_3RDPARTY_HEADER_TARGETS)
        if(TARGET ${dep})
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(def TARGET ${dep} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
            if(def)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
            endif()
        endif()
    endforeach()
endfunction()

# Convenience function to set important target properties
function(open3d_set_global_properties target)
    # Libraries need to be compiled with position independent code
    get_target_property(target_type ${target} TYPE)
    if (NOT target_type STREQUAL "EXECUTABLE")
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    # Tell CMake we want a compiler that supports C++11 features
    target_compile_features(${target} PUBLIC cxx_variadic_templates cxx_constexpr)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}>
    )
    if(BUILD_CUDA_MODULE)
        target_compile_definitions(${target} PRIVATE BUILD_CUDA_MODULE)
    endif()
    if(GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    endif()
    if(ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(${target} PRIVATE HEADLESS_RENDERING)
    endif()
    if(WIN32)
        target_compile_definitions(${target} PRIVATE
            WINDOWS
            _CRT_SECURE_NO_DEPRECATE
            _CRT_NONSTDC_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
            GLEW_STATIC
        )
        if(MSVC)
            target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES _ENABLE_EXTENDED_ALIGNED_STORAGE)
            # Multi-thread compile, two ways to enable
            # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
            # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
            # OPtion 2, at configure time: add /MP flag, no need to use Option 1
            # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
            #
            # /bigobj is enabled by default. On Windows, the build type (Debug, Release)
            # is set at build time, not config time, since CMAKE_BUILD_TYPE is not
            # supported on Windows.
            # ref: https://stackoverflow.com/a/15110915
            # ref: https://stackoverflow.com/a/19026241
            target_compile_options(${target} PRIVATE /EHsc /wd4522 /WX /bigobj)
        endif()
        if (STATIC_WINDOWS_RUNTIME)
            set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        endif()
    elseif(APPLE)
        target_compile_definitions(${target} PRIVATE UNIX APPLE)
        target_compile_options(${target} PRIVATE "-Wno-deprecated-declarations")
    elseif(UNIX)
        target_compile_definitions(${target} PRIVATE UNIX)
        target_compile_options(${target} PRIVATE "-Wno-deprecated-declarations" $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>)
    endif()
endfunction()

macro(add_source_group module_name)
    file(GLOB MODULE_HEADER_FILES "${module_name}/*.h")
    source_group("Header Files\\${module_name}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${module_name}/*.cpp")
    source_group("Source Files\\${module_name}" FILES ${MODULE_SOURCE_FILES})
endmacro()

# Open3D library
add_subdirectory(src)

# Examples
add_subdirectory(examples)

# Install CMake configuration files
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${Open3D_INSTALL_CMAKE_DIR})
export(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::)

# `make check-style` errors if styling is not compliant
add_custom_target(check-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/scripts/check-style.cmake
)

# `make apply-style` runs clang-format to format all source code
add_custom_target(apply-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/scripts/apply-style.cmake
)
