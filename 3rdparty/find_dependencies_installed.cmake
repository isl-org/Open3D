# Minimal 3rd-party setup for OPEN3D_USE_INSTALLED_LIBRARY=ON.
# find_package(Open3D) has already imported Open3D::Open3D and public
# interface targets (e.g. Open3D::3rdparty_eigen3, Open3D::tbb). This file
# only adds what is still needed to compile pybind and optional ML ops:
# pybind11, header-only deps not exported from shared devel packages, and
# libc++ for Filament-linked libOpen3D on Linux.

message(STATUS "Using slim 3rdparty setup for OPEN3D_USE_INSTALLED_LIBRARY")

# Pybind11 (not part of the C++ devel package)
if(BUILD_PYTHON_MODULE)
    if(USE_SYSTEM_PYBIND11)
        find_package(pybind11)
    endif()
    if(NOT USE_SYSTEM_PYBIND11 OR NOT TARGET pybind11::module)
        set(USE_SYSTEM_PYBIND11 OFF)
        include(${Open3D_3RDPARTY_DIR}/pybind11/pybind11.cmake)
    endif()
endif()

# Nanoflann is PRIVATE in shared builds, so it is not in Open3DTargets.cmake.
# ML ops still need the headers when BUILD_*_OPS=ON.
if(BUILD_PYTORCH_OPS OR BUILD_TENSORFLOW_OPS)
    if(NOT TARGET Open3D::3rdparty_nanoflann)
        if(USE_SYSTEM_NANOFLANN)
            open3d_find_package_3rdparty_library(3rdparty_nanoflann
                PACKAGE nanoflann
                VERSION 1.5.0
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
        endif()
    endif()
endif()

# CUDA header libs for ML ops (same as full find_dependencies, gated on ops)
if(BUILD_CUDA_MODULE AND (BUILD_PYTORCH_OPS OR BUILD_TENSORFLOW_OPS))
    if(CUDAToolkit_VERSION VERSION_LESS "11.0" AND NOT TARGET Open3D::3rdparty_cub)
        include(${Open3D_3RDPARTY_DIR}/cub/cub.cmake)
        open3d_import_3rdparty_library(3rdparty_cub
            INCLUDE_DIRS ${CUB_INCLUDE_DIRS}
            DEPENDS      ext_cub
        )
    endif()
    if(NOT TARGET Open3D::3rdparty_cutlass)
        if(USE_SYSTEM_CUTLASS)
            find_path(3rdparty_cutlass_INCLUDE_DIR NAMES cutlass/cutlass.h)
            if(3rdparty_cutlass_INCLUDE_DIR)
                add_library(3rdparty_cutlass INTERFACE)
                target_include_directories(3rdparty_cutlass INTERFACE
                    ${3rdparty_cutlass_INCLUDE_DIR})
                add_library(Open3D::3rdparty_cutlass ALIAS 3rdparty_cutlass)
            else()
                set(USE_SYSTEM_CUTLASS OFF)
            endif()
        endif()
        if(NOT USE_SYSTEM_CUTLASS)
            include(${Open3D_3RDPARTY_DIR}/cutlass/cutlass.cmake)
            open3d_import_3rdparty_library(3rdparty_cutlass
                INCLUDE_DIRS ${CUTLASS_INCLUDE_DIRS}
                DEPENDS      ext_cutlass
            )
        endif()
    endif()
endif()

# SYCL ML ops are compiled in this slim configuration, so recreate the
# interface targets normally supplied by the full dependency configuration.
if(BUILD_SYCL_MODULE AND (BUILD_PYTORCH_OPS OR BUILD_TENSORFLOW_OPS))
    if(NOT TARGET Open3D::3rdparty_sycl)
        add_library(3rdparty_sycl INTERFACE)
        target_link_libraries(3rdparty_sycl INTERFACE
            $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<LINK_LANGUAGE:ISPC>>>:sycl>)
        target_link_options(3rdparty_sycl INTERFACE
            $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<LINK_LANGUAGE:ISPC>>>:-fsycl -fsycl-targets=${OPEN3D_SYCL_TARGETS}>)
        if(OPEN3D_SYCL_TARGET_BACKEND_OPTIONS)
            target_link_options(3rdparty_sycl INTERFACE
                $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<LINK_LANGUAGE:ISPC>>>:-Xs ${OPEN3D_SYCL_TARGET_BACKEND_OPTIONS}>)
        endif()
        add_library(Open3D::3rdparty_sycl ALIAS 3rdparty_sycl)
    endif()

    if(NOT TARGET Open3D::3rdparty_sycl_tla)
        include(${Open3D_3RDPARTY_DIR}/sycl_tla/sycl_tla.cmake)
        open3d_import_3rdparty_library(3rdparty_sycl_tla
            INCLUDE_DIRS ${SYCL_TLA_INCLUDE_DIRS}
            DEPENDS      ext_sycl_tla
        )
        target_compile_definitions(3rdparty_sycl_tla INTERFACE
            CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
        if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
            target_link_options(3rdparty_sycl_tla INTERFACE
                "-Xspirv-translator"
                "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
        endif()
    endif()
endif()

# Ensure TBB::tbb exists for PYTHON_EXTRA_LIBRARIES and ML ops link lines.
# Devel packages export Open3D::tbb; Open3DConfig may also find_dependency(TBB).
if(NOT TARGET TBB::tbb)
    find_package(TBB QUIET)
endif()
if(NOT TARGET TBB::tbb AND TARGET Open3D::tbb)
    add_library(TBB::tbb ALIAS Open3D::tbb)
endif()
if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR
        "TBB::tbb not found. Install TBB or use an Open3D devel package that "
        "exports Open3D::tbb / find_dependency(TBB).")
endif()

# Filament-linked libOpen3D needs matching libc++/libc++abi in the wheel on Linux.
if(BUILD_GUI AND BUILD_PYTHON_MODULE AND UNIX AND NOT APPLE)
    if(NOT CPP_LIBRARY OR NOT CPPABI_LIBRARY)
        message(STATUS "Searching /usr/lib/llvm-[7..19]/lib/ for libc++ and libc++abi")
        foreach(llvm_ver RANGE 7 19)
            set(llvm_lib_dir "/usr/lib/llvm-${llvm_ver}/lib")
            find_library(CPP_LIBRARY    c++ PATHS ${llvm_lib_dir} NO_DEFAULT_PATH)
            find_library(CPPABI_LIBRARY c++abi PATHS ${llvm_lib_dir} NO_DEFAULT_PATH)
            if(CPP_LIBRARY AND CPPABI_LIBRARY)
                message(STATUS "Found libc++ in ${llvm_lib_dir}")
                break()
            endif()
            unset(CPP_LIBRARY CACHE)
            unset(CPPABI_LIBRARY CACHE)
        endforeach()
    endif()
    if(NOT CPP_LIBRARY OR NOT CPPABI_LIBRARY)
        message(WARNING
            "libc++/libc++abi not found; GUI wheels may fail to load Filament "
            "symbols from the installed libOpen3D.")
    endif()
endif()

# Keep link-helper lists defined (empty) so open3d_link_3rdparty_libraries is safe
# if any remaining object targets call it.
set(Open3D_3RDPARTY_PUBLIC_TARGETS "")
set(Open3D_3RDPARTY_PRIVATE_TARGETS "")
set(Open3D_3RDPARTY_HEADER_TARGETS "")
set(BUILD_WEBRTC_COMMENT "//")
