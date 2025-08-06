cmake_minimum_required(VERSION 3.24)
# If you're using Ubuntu 20.04, we suggest you install the latest CMake from the
# official repository https://apt.kitware.com/.
# CMake 3.24+ is required for CUDA native arch selection
# CMake 3.22+ is required by Assimp v5.4.2
# CMake 3.20+ is required to detect IntelLLVM compiler for SYCL

# CMAKE_HOST_SYSTEM_PROCESSOR is only available after calling project(),
# which depends on ${OPEN3D_VERSION}, which depends on ${DEVELOPER_BUILD}.
if(UNIX AND NOT APPLE)
    execute_process(COMMAND uname -m
        OUTPUT_VARIABLE PROCESSOR_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PROCESSOR_ARCH STREQUAL "aarch64")
        set(LINUX_AARCH64 TRUE)
    endif()
endif()
if(APPLE)
    execute_process(COMMAND uname -m
        OUTPUT_VARIABLE PROCESSOR_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PROCESSOR_ARCH STREQUAL "arm64")
        set(APPLE_AARCH64 TRUE)
        set (CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING
            "Minimum OS X deployment version" FORCE)
    else()
        set (CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING
            "Minimum OS X deployment version" FORCE)
    endif()
endif()

include(CMakeDependentOption)

# Open3D build options
option(BUILD_SHARED_LIBS          "Build shared libraries"                   OFF)
option(BUILD_EXAMPLES             "Build Open3D examples programs"           ON )
option(BUILD_UNIT_TESTS           "Build Open3D unit tests"                  OFF)
option(BUILD_BENCHMARKS           "Build the micro benchmarks"               OFF)
option(BUILD_PYTHON_MODULE        "Build the python module"                  ON )
option(BUILD_CUDA_MODULE          "Build the CUDA module"                    OFF)
option(BUILD_WITH_CUDA_STATIC     "Build with static CUDA libraries"         ON )
option(BUILD_COMMON_CUDA_ARCHS    "Build for common CUDA GPUs (for release)" OFF)
if (WIN32)   # Causes CUDA runtime error on Windows (See issue #6555)
    option(ENABLE_CACHED_CUDA_MANAGER "Enable cached CUDA memory manager"    OFF)
else()
    option(ENABLE_CACHED_CUDA_MANAGER "Enable cached CUDA memory manager"    ON )
endif()
if(NOT LINUX_AARCH64 AND NOT APPLE_AARCH64)
    option(BUILD_ISPC_MODULE      "Build the ISPC module"                    ON )
else()
    option(BUILD_ISPC_MODULE      "Build the ISPC module"                    OFF)
endif()
option(BUILD_COMMON_ISPC_ISAS     "Build for common ISPC ISAs (for release)" OFF)
option(BUILD_GUI                  "Builds new GUI"                           ON )
option(WITH_OPENMP                "Use OpenMP multi-threading"               ON )
option(WITH_IPP                "Use Intel Integrated Performance Primitives" ON )
option(ENABLE_HEADLESS_RENDERING  "Use OSMesa for headless rendering"        OFF)
if(BUILD_SHARED_LIBS)
    option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime"      OFF)
else()
    option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime"      ON )
endif()
option(BUILD_SYCL_MODULE          "Build SYCL module with Intel oneAPI"      OFF)
if(BUILD_SYCL_MODULE)
    set(OPEN3D_SYCL_TARGETS "spir64" CACHE STRING
    "SYCL targets: spir64 for JIT, or another for AOT compilation. See https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md."
)
    set(OPEN3D_SYCL_TARGET_BACKEND_OPTIONS "" CACHE STRING
    "SYCL target backend options, e.g. to compile for a specific device. See https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md."
)
    set(BUILD_ISPC_MODULE OFF CACHE BOOL "Build the ISPC module" FORCE)
    set(BUILD_CUDA_MODULE OFF CACHE BOOL "Build the CUDA module" FORCE)
endif()
option(GLIBCXX_USE_CXX11_ABI      "Set -D_GLIBCXX_USE_CXX11_ABI=1"           ON )
option(ENABLE_SYCL_UNIFIED_SHARED_MEMORY "Enable SYCL unified shared memory" OFF)
if(BUILD_GUI AND (WIN32 OR UNIX AND NOT LINUX_AARCH64 AND NOT APPLE_AARCH64))
    option(BUILD_WEBRTC           "Build WebRTC visualizer"                  ON )
else()
    option(BUILD_WEBRTC           "Build WebRTC visualizer"                  OFF)
endif()
option(BUILD_JUPYTER_EXTENSION    "Build Jupyter, requires BUILD_WEBRTC=ON"  OFF)

# 3rd-party build options
if(LINUX_AARCH64 OR APPLE_AARCH64)
    option(USE_BLAS               "Use BLAS/LAPACK instead of MKL"           ON )
else()
    option(USE_BLAS               "Use BLAS/LAPACK instead of MKL"           OFF)
endif()
if(USE_BLAS)
    option(USE_SYSTEM_BLAS        "Use system pre-installed openblas"        OFF)
else()
    option(USE_SYSTEM_BLAS        "Use system pre-installed openblas"        ON )
endif()
option(USE_SYSTEM_ASSIMP          "Use system pre-installed assimp"          OFF)
if (APPLE)
    option(USE_SYSTEM_CURL            "Use system pre-installed curl"         ON)
else()
    option(USE_SYSTEM_CURL            "Use system pre-installed curl"        OFF)
endif()
option(USE_SYSTEM_CUTLASS         "Use system pre-installed cutlass"         OFF)
option(USE_SYSTEM_EIGEN3          "Use system pre-installed eigen3"          OFF)
option(USE_SYSTEM_EMBREE          "Use system pre-installed Embree"          OFF)
option(USE_SYSTEM_FILAMENT        "Use system pre-installed filament"        OFF)
option(USE_SYSTEM_FMT             "Use system pre-installed fmt"             OFF)
option(USE_SYSTEM_GLEW            "Use system pre-installed glew"            OFF)
option(USE_SYSTEM_GLFW            "Use system pre-installed glfw"            OFF)
option(USE_SYSTEM_GOOGLETEST      "Use system pre-installed Googletest"      OFF)
option(USE_SYSTEM_IMGUI           "Use system pre-installed imgui"           OFF)
option(USE_SYSTEM_JPEG            "Use system pre-installed jpeg"            OFF)
option(USE_SYSTEM_JSONCPP         "Use system pre-installed jsoncpp"         OFF)
option(USE_SYSTEM_LIBLZF          "Use system pre-installed liblzf"          OFF)
option(USE_SYSTEM_MSGPACK         "Use system pre-installed msgpack"         OFF)
option(USE_SYSTEM_NANOFLANN       "Use system pre-installed nanoflann"       OFF)
option(USE_SYSTEM_OPENSSL         "Use system pre-installed OpenSSL"         OFF)
option(USE_SYSTEM_PNG             "Use system pre-installed png"             OFF)
option(USE_SYSTEM_PYBIND11        "Use system pre-installed pybind11"        OFF)
option(USE_SYSTEM_QHULLCPP        "Use system pre-installed qhullcpp"        OFF)
option(USE_SYSTEM_STDGPU          "Use system pre-installed stdgpu"          OFF)
option(USE_SYSTEM_TBB             "Use system pre-installed TBB"             OFF)
option(USE_SYSTEM_TINYGLTF        "Use system pre-installed tinygltf"        OFF)
option(USE_SYSTEM_TINYOBJLOADER   "Use system pre-installed tinyobjloader"   OFF)
option(USE_SYSTEM_VTK             "Use system pre-installed VTK"             OFF)
option(USE_SYSTEM_ZEROMQ          "Use system pre-installed ZeroMQ"          OFF)
if(LINUX_AARCH64 OR APPLE_AARCH64)
    option(BUILD_VTK_FROM_SOURCE      "Build VTK from source"                ON )
else()
    option(BUILD_VTK_FROM_SOURCE      "Build VTK from source"                OFF)
endif()
if(LINUX_AARCH64)
    option(BUILD_FILAMENT_FROM_SOURCE "Build filament from source"           ON )
else()
    option(BUILD_FILAMENT_FROM_SOURCE "Build filament from source"           OFF)
endif()

option(PREFER_OSX_HOMEBREW        "Prefer Homebrew libs over frameworks"     ON )
option(WITH_MINIZIP               "Enable MiniZIP"                           OFF)

# Sensor options
option(BUILD_LIBREALSENSE         "Build support for Intel RealSense camera" OFF)
option(USE_SYSTEM_LIBREALSENSE    "Use system pre-installed librealsense"    OFF)
option(BUILD_AZURE_KINECT         "Build support for Azure Kinect sensor"    OFF)

# ML library options
option(BUILD_TENSORFLOW_OPS       "Build ops for TensorFlow"                 OFF)
option(BUILD_PYTORCH_OPS          "Build ops for PyTorch"                    OFF)
option(BUNDLE_OPEN3D_ML           "Includes the Open3D-ML repo in the wheel" OFF)

# Release build options
option(DEVELOPER_BUILD      "Add +commit_hash to the project version number" ON )
if (NOT DEVELOPER_BUILD)
    if (NOT BUILD_COMMON_CUDA_ARCHS)
        set(BUILD_COMMON_CUDA_ARCHS ON CACHE BOOL "Build for common CUDA GPUs (for release)" FORCE)
        message(WARNING "Setting BUILD_COMMON_CUDA_ARCHS=ON since DEVELOPER_BUILD is OFF.")
    endif()
endif()

# Default build type on single-config generators.
# For multi-config generators (e.g. Visual Studio), CMAKE_CONFIGURATION_TYPES
# will be set, and we don't specify a default CMAKE_BUILD_TYPE.
# https://blog.kitware.com/cmake-and-the-default-build-type/
if(NOT CMAKE_CONFIGURATION_TYPES)
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "Setting build type to Release as none was specified.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
        # Set the possible values of build type for cmake-gui.
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
    message(STATUS "CMAKE_BUILD_TYPE is set to ${CMAKE_BUILD_TYPE}.")
endif()

find_program(CCACHE "ccache")
if (CCACHE)
    message(STATUS "ccache found at ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    if(BUILD_CUDA_MODULE)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
    endif()
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)  # URL contents timestamped by download time
endif()
# In ExternalProject_Add, if OPEN3D_THIRD_PARTY_DOWNLOAD_DIR is specified, CMake will
# use this directory to cache downloaded 3rd party dependencies and automatically skip
# downloading from the Internet if the files are available. This is only supported by
# a limited number of 3rd party libraries.
set(OPEN3D_THIRD_PARTY_DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty_downloads"
    CACHE PATH "Third-party download directory for caching.")
message(STATUS "Downloading third-party dependencies to ${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}")

set(FILAMENT_PRECOMPILED_ROOT "" CACHE PATH "Path to precompiled Filament library (used if BUILD_FILAMENT_FROM_SOURCE=OFF)")

if (PREFER_OSX_HOMEBREW)
    set(CMAKE_FIND_FRAMEWORK LAST)
    set(CMAKE_FIND_APPBUNDLE LAST)
endif()

# Set OpenGL policy
if(NOT USE_SYSTEM_GLFW)
    cmake_policy(SET CMP0072 OLD)
endif()
cmake_policy(GET CMP0072 CMP0072_VALUE)

# Catch a few incompatible build options
if ((LINUX_AARCH64 OR APPLE_AARCH64) AND BUILD_ISPC_MODULE)
    message(FATAL_ERROR "ISPC module is not yet supported on ARM Linux")
endif()
if (LINUX_AARCH64 AND NOT BUILD_FILAMENT_FROM_SOURCE)
    message(FATAL_ERROR "ARM CPU detected, you must set BUILD_FILAMENT_FROM_SOURCE=ON.")
endif()
if ((LINUX_AARCH64 OR APPLE_AARCH64) AND NOT USE_BLAS)
    message(FATAL_ERROR "ARM CPU detected, you must set USE_BLAS=ON.")
endif()
if (APPLE AND ENABLE_HEADLESS_RENDERING)
    message(WARNING "Headless rendering is not supported on Mac OS")
    set(ENABLE_HEADLESS_RENDERING OFF)
endif()
if(ENABLE_HEADLESS_RENDERING AND BUILD_GUI)
    message(WARNING "Headless rendering disables the Open3D GUI")
    set(BUILD_GUI OFF)
endif()
if(ENABLE_HEADLESS_RENDERING AND (USE_SYSTEM_GLEW OR USE_SYSTEM_GLFW))
    message(WARNING "Headless rendering requires customized GLEW and GLFW builds")
    set(USE_SYSTEM_GLEW OFF)
    set(USE_SYSTEM_GLFW OFF)
endif()
if(BUNDLE_OPEN3D_ML AND NOT (BUILD_TENSORFLOW_OPS OR BUILD_PYTORCH_OPS))
    message(SEND_ERROR "3DML depends on TensorFlow or PyTorch Ops. Enable them with -DBUILD_TENSORFLOW_OPS=ON or -DBUILD_PYTORCH_OPS=ON")
endif()
if(BUILD_WEBRTC AND LINUX_AARCH64)
    message(FATAL_ERROR "BUILD_WEBRTC=ON is not yet supported on ARM Linux")
endif()
if(BUILD_WEBRTC AND NOT BUILD_GUI)
    message(FATAL_ERROR "BUILD_WEBRTC=ON requires BUILD_GUI=ON")
endif()
if(BUILD_JUPYTER_EXTENSION AND NOT BUILD_WEBRTC)
    # BUILD_JUPYTER_EXTENSION transitively depends on BUILD_GUI
    message(FATAL_ERROR "BUILD_JUPYTER_EXTENSION=ON requires BUILD_WEBRTC=ON")
endif()
if(BUILD_JUPYTER_EXTENSION AND NOT BUILD_PYTHON_MODULE)
    message(FATAL_ERROR "BUILD_JUPYTER_EXTENSION=ON requires BUILD_PYTHON_MODULE=ON")
endif()

# Parse Open3D version number
file(STRINGS "cpp/open3d/version.txt" OPEN3D_VERSION_READ)
foreach(ver ${OPEN3D_VERSION_READ})
    if (ver MATCHES "OPEN3D_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(OPEN3D_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(OPEN3D_VERSION_DEVHASH "")
if(DEVELOPER_BUILD)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" log --pretty=format:%h -n 1
        OUTPUT_VARIABLE GIT_REV)
    if (GIT_REV)
        set(OPEN3D_VERSION_DEVHASH "+${GIT_REV}")
    endif()
endif()
string(CONCAT OPEN3D_VERSION
    "${OPEN3D_VERSION_MAJOR}"
    ".${OPEN3D_VERSION_MINOR}"
    ".${OPEN3D_VERSION_PATCH}"
)
set(OPEN3D_VERSION_FULL "${OPEN3D_VERSION}${OPEN3D_VERSION_DEVHASH}" CACHE
    STRING "Open3D full version.")
set(OPEN3D_ABI_VERSION "${OPEN3D_VERSION_MAJOR}.${OPEN3D_VERSION_MINOR}" CACHE
    STRING "Open3D ABI version / SOVERSION (for releases only).")
# Set additional info
set(PROJECT_EMAIL       "open3d@intel.com")
set(PROJECT_DOCS        "https://www.open3d.org/docs")
set(PROJECT_CODE        "https://github.com/isl-org/Open3D")
set(PROJECT_ISSUES      "https://github.com/isl-org/Open3D/issues")

project(Open3D
    VERSION ${OPEN3D_VERSION}
    # Set PROJECT_DESCRIPTION
    DESCRIPTION "Open3D: A Modern Library for 3D Data Processing."
    # Set PROJECT_HOMEPAGE_URL
    HOMEPAGE_URL "https://www.open3d.org"
    LANGUAGES C CXX)
message(STATUS "Open3D ${OPEN3D_VERSION_FULL}")

# Check SYCL compatiblility
if (BUILD_SYCL_MODULE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(FATAL_ERROR "BUILD_SYCL_MODULE requires IntelLLVM (DPC++) compiler, "
                        "but got CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID} "
                        "and CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}.")
endif()
if (BUILD_SYCL_MODULE AND (NOT UNIX OR APPLE))
    message(FATAL_ERROR "Open3D SYCL support is only available on Linux")
endif()
if(BUILD_SYCL_MODULE AND NOT GLIBCXX_USE_CXX11_ABI)
    message(FATAL_ERROR "BUILD_SYCL_MODULE=ON requires GLIBCXX_USE_CXX11_ABI=ON")
endif()
if(BUILD_SYCL_MODULE AND BUILD_CUDA_MODULE)
    message(FATAL_ERROR "BUILD_SYCL_MODULE and BUILD_SYCL_MODULE cannot be on at the same time for now.")
endif()

# Global flag to set CXX standard.
# This does not affect 3rd party libraries.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)   # Improved compatibility

# Suppress warnings for deprecated C++17 functions (stdgpu->thrust with CUDA 11 for MSVC).
add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING>)
# CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cmake")

# Setup Python executable
if(NOT DEFINED Python3_FIND_REGISTRY)
    # Only consider PATH variable on Windows by default
    set(Python3_FIND_REGISTRY NEVER)
endif()
# Requires Python 3.6+
find_package(Python3 3.6
             COMPONENTS Interpreter Development)
if (Python3_FOUND)
    # Setup PYTHON_EXECUTABLE for 3rdparty modules
    # which still use the deprecated find_package(PythonInterp)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE STRING
        "Deprecated path to the Python executable (for 3rdparty only)" FORCE)
else()
    if (BUILD_PYTHON_MODULE)
        message(FATAL_ERROR "BUILD_PYTHON_MODULE=ON requires Python >= 3.6. Please ensure it is in PATH.")
    endif()
endif()

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

# Set installation paths
if(UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(Open3D_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(Open3D_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(Open3D_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    # Put resources in */share/
    set(Open3D_INSTALL_RESOURCE_DIR "${CMAKE_INSTALL_DATADIR}")
    set(Open3D_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(Open3D_INSTALL_INCLUDE_DIR include)
    set(Open3D_INSTALL_BIN_DIR bin)
    set(Open3D_INSTALL_LIB_DIR lib)
    # Put resources in */bin, with executables / DLLs
    set(Open3D_INSTALL_RESOURCE_DIR bin)
    set(Open3D_INSTALL_CMAKE_DIR CMake)
endif()

# Put build results in some predictable places
# The $<CONFIG> generator expression makes sure that XCode or Visual Studio do not
# append additional path components, as we need to know *exactly* where the build results
# end up.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Global Security options (including 3rd party code)
# Add -fPIC for library and -fPIE for executable to compiler and linker. Does not add -pie !
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Explicitly specify the preference of using -pthread over -lpthread.
# This must be defined here since CUDA calls find_package(Threads) internally.
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Overwrites property for Thread::Threads in find_package(Threads)
# For CUDA, "-pthread" is replaced with "-Xcompiler -pthread" (CMake's default)
# For ISPC, "-pthread" is disabled
macro(open3d_patch_findthreads_module_)
    if(TARGET Threads::Threads AND THREADS_HAVE_PTHREAD_ARG)
        set_property(TARGET Threads::Threads
                     PROPERTY INTERFACE_COMPILE_OPTIONS
                     "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler -pthread>"
                     "$<$<AND:$<NOT:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>>,$<NOT:$<COMPILE_LANGUAGE:ISPC>>>:-pthread>")
    endif()
endmacro()
cmake_language(EVAL CODE "cmake_language(DEFER CALL open3d_patch_findthreads_module_)")

# Build CUDA module by default if CUDA is available
if(BUILD_CUDA_MODULE)
    # Suppress nvcc unsupported compiler error for MSVC 2022 with CUDA 11.7 to 12.4
    # https://forums.developer.nvidia.com/t/problems-with-latest-vs2022-update/294150/12
    if (MSVC AND MSVC_VERSION VERSION_LESS_EQUAL "1949")
        # Set this before any CUDA checks
        add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
        set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler" CACHE STRING "Additional flags for nvcc" FORCE)
        message(WARNING "Using --allow-unsupported-compiler flag for nvcc with MSVC 2022. "
        "Set $Env:NVCC_PREPEND_FLAGS='--allow-unsupported-compiler' if nvcc still fails.")
    endif()
    if (CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "Building with user-provided CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        if(BUILD_COMMON_CUDA_ARCHS)
            # Build with all supported architectures for previous 2 generations and
            # M0 (minor=0) architectures for previous generations (including
            # deprecated). Note that cubin for M0 runs on GPUs with architecture Mx.
            # This is a tradeoff between binary size / build time and runtime on
            # older architectures. See:
            # https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#building-for-maximum-compatibility
            # https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html#application-compatibility-on-ampere
            # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
            find_package(CUDAToolkit REQUIRED)
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90)    # Turing, Ampere, Ada Lovelace, Hopper
            elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
                set(CMAKE_CUDA_ARCHITECTURES 70-real 75-real 80-real 86)            # Volta, Turing, Ampere
            elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
                set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 72-real 75-real 80)    # Pascal, Volta, Turing, Ampere
            else()
                set(CMAKE_CUDA_ARCHITECTURES 30-real 50-real 60-real 70-real 75)    # Kepler, Maxwell, Pascal, Turing
            endif()
            message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            execute_process(COMMAND nvidia-smi RESULT_VARIABLE NVIDIA_CHECK OUTPUT_QUIET)
            if (NVIDIA_CHECK EQUAL 0)
                message(STATUS "Building with native CUDA architecture.")
                set(CMAKE_CUDA_ARCHITECTURES native)
            else()
                message(WARNING "No CUDA GPU detected. Building with CMake default CUDA architecture.")
            endif()
        endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.5")
        message(FATAL_ERROR "CUDA 11.4 and older are not supported. Please upgrade to CUDA 11.5 or newer.")
    endif()
endif()

# ISPC language emulation support
include(Open3DISPC)

if (CMAKE_ISPC_COMPILER_LOADED OR (CMAKE_GENERATOR MATCHES "Make" OR CMAKE_GENERATOR MATCHES "Ninja"))
    option(ISPC_USE_LEGACY_EMULATION "Use legacy ISPC language emulation over first-class CMake support" OFF)
else()
    option(ISPC_USE_LEGACY_EMULATION "Use legacy ISPC language emulation over first-class CMake support" ON)
endif()
mark_as_advanced(ISPC_USE_LEGACY_EMULATION)
option(ISPC_PRINT_LEGACY_COMPILE_COMMANDS "Prints legacy compile commands on CMake configuration time" ON)
mark_as_advanced(ISPC_PRINT_LEGACY_COMPILE_COMMANDS)

# Build ISPC module by default if ISPC is available
if (BUILD_ISPC_MODULE)
    include(Open3DFetchISPCCompiler)
    open3d_fetch_ispc_compiler()

    include(Open3DMakeISPCInstructionSets)
    open3d_make_ispc_instruction_sets(ISPC_ISAS)
    set(CMAKE_ISPC_INSTRUCTION_SETS ${ISPC_ISAS})

    message(STATUS "Using ISPC instruction sets: ${CMAKE_ISPC_INSTRUCTION_SETS}")

    open3d_ispc_enable_language(ISPC)

    if (CMAKE_ISPC_COMPILER_ID STREQUAL "Intel" AND CMAKE_ISPC_COMPILER_VERSION VERSION_LESS "1.16")
        message(FATAL_ERROR "ISPC 1.15 and older are not supported. Please upgrade to ISPC 1.16 or newer.")
    endif()

    if (NOT CMAKE_ISPC_COMPILER_ID)
        message(FATAL_ERROR "Unknown ISPC compiler.")
    endif()
endif()

# OS specific settings
if(WIN32)
    # Set oldest supported Windows version to Windows 10 (10.0, 0x0A00)
    # https://learn.microsoft.com/en-us/cpp/porting/modifying-winver-and-win32-winnt?view=msvc-170
    add_compile_definitions(WINVER=0x0A00 _WIN32_WINNT=0x0A00)
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explicitly export symbols.
    #       Then, we could use -fvisibility=hidden for Linux as well
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)
        # MSVC standards-compliant mode: Error in PoissonRecon 3rd party header
        # add_compile_options($<$<COMPILE_LANGUAGE:CXX>:"/permissive-">)
        # Make sure we don't hit the 65535 object member limit with MSVC
        #
        # /bigobj allows object files with more than 65535 members
        # /Ob2 enables function inlining, because MSVC is particularly
        # verbose with inline members
        #
        # See: https://github.com/tensorflow/tensorflow/pull/10962
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Ob2>")
    endif()
    if (STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

# Folder view for project files
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

macro(add_source_group module_name)
    file(GLOB MODULE_HEADER_FILES "${module_name}/*.h")
    source_group("Header Files\\${module_name}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${module_name}/*.cpp")
    source_group("Source Files\\${module_name}" FILES ${MODULE_SOURCE_FILES})
    file(GLOB MODULE_ISPC_HEADER_FILES "${module_name}/*.isph")
    source_group("Header Files\\ISPC" FILES ${MODULE_ISPC_HEADER_FILES})
    file(GLOB MODULE_ISPC_SOURCE_FILES "${module_name}/*.ispc")
    source_group("Source Files\\ISPC" FILES ${MODULE_ISPC_SOURCE_FILES})
    file(GLOB MODULE_SHADER_FILES "${module_name}/*.glsl")
    source_group("Source Files\\Shader\\GLSL" FILES ${MODULE_SHADER_FILES})
    file(GLOB MODULE_MATERIAL_FILES "${module_name}/*.mat")
    source_group("Source Files\\Material" FILES ${MODULE_MATERIAL_FILES})
endmacro()

if (LINUX_AARCH64)
# Fix for ImportError: ... /pybind.cpython-310-aarch64-linux-gnu.so: cannot allocate memory in static TLS block
# https://bugs.launchpad.net/ubuntu/+source/mysql-8.0/+bug/1889851
    add_compile_options("-ftls-model=global-dynamic")
endif()

# Include convenience functions
include(Open3DLink3rdpartyLibraries)
include(Open3DSetGlobalProperties)
include(Open3DShowAndAbortOnWarning)
include(Open3DSYCLTargetSources)

# Enumerate all third-party libraries which we need later
# This creates the necessary targets and sets the
# Open3D_3RDPARTY_*_TARGETS variables we use in open3d_link_3rdparty_libraries
include(3rdparty/find_dependencies.cmake)

# Open3D library
add_subdirectory(cpp)

# Examples
add_subdirectory(examples)

# Documentation
add_subdirectory(docs)

# Install CMake configuration files
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${Open3D_INSTALL_CMAKE_DIR})
export(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::)

if (Python3_EXECUTABLE)
    # `make check-style` checks style for c++/cuda/python/ipynb files
    add_custom_target(check-style
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/util/check_style.py
        COMMENT "Python executable used for style check: ${Python3_EXECUTABLE}."
    )

    # `make apply-style` applies style for c++/cuda/python/ipynb files
    add_custom_target(apply-style
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/util/check_style.py --apply
        COMMENT "Python executable used for style check: ${Python3_EXECUTABLE}."
    )
endif()

include(Open3DPackaging)

# `make check-cpp-style` checks style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(check-cpp-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DAPPLY=OFF
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/check_cpp_style.cmake
)

# `make apply-cpp-style` applies style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(apply-cpp-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DAPPLY=ON
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/check_cpp_style.cmake
)

include(Open3DPrintConfigurationSummary)
open3d_print_configuration_summary()