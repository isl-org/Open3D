# Find the PyTorch root and use the provided cmake module
#
# The following variables will be set:
# - Pytorch_FOUND
# - Pytorch_VERSION
# - Pytorch_ROOT
# - Pytorch_DEFINITIONS
#
# This script will call find_package( Torch ) which will define:
# - TORCH_FOUND
# - TORCH_INCLUDE_DIRS
# - TORCH_LIBRARIES
# - TORCH_CXX_FLAGS
#
# and import the target 'torch'.

# "80-real" to "8.0" and "80" to "8.0+PTX":
macro(translate_arch_string input output)
    if("${input}" MATCHES "[0-9]+-real")
        string(REGEX REPLACE "([1-9])([0-9])-real" "\\1.\\2" version "${input}")
    elseif("${input}" MATCHES "([0-9]+)")
        string(REGEX REPLACE "([1-9])([0-9])" "\\1.\\2+PTX" version "${input}")
    elseif("${input}" STREQUAL "native")
        set(version "Auto")
    else()
        message(FATAL_ERROR "Invalid architecture string: ${input}")
    endif()
    set(${output} "${version}")
endmacro()

if(NOT Pytorch_FOUND)
    # Searching for pytorch requires the python executable
    if (NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "Python 3 not found in top level file")
    endif()

    message(STATUS "Getting PyTorch properties ...")

    set(PyTorch_FETCH_PROPERTIES
        "import os"
        "import torch"
        "print(torch.__version__, end=';')"
        "print(os.path.dirname(torch.__file__), end=';')"
        "print(torch._C._GLIBCXX_USE_CXX11_ABI, end=';')"
    )
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "-c" "${PyTorch_FETCH_PROPERTIES}"
        OUTPUT_VARIABLE PyTorch_PROPERTIES
    )

    list(GET PyTorch_PROPERTIES 0 Pytorch_VERSION)
    list(GET PyTorch_PROPERTIES 1 Pytorch_ROOT)
    list(GET PyTorch_PROPERTIES 2 Pytorch_CXX11_ABI)

    unset(PyTorch_FETCH_PROPERTIES)
    unset(PyTorch_PROPERTIES)

    if(BUILD_CUDA_MODULE)
    # Using CUDA 12.x and Pytorch <2.4 gives the error "Unknown CUDA Architecture Name 9.0a in CUDA_SELECT_NVCC_ARCH_FLAGS".
    # As a workaround we explicitly set TORCH_CUDA_ARCH_LIST
        set(TORCH_CUDA_ARCH_LIST "")
        foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            translate_arch_string("${arch}" ptarch)
            list(APPEND TORCH_CUDA_ARCH_LIST "${ptarch}")
        endforeach()
        message(STATUS "Using top level CMAKE_CUDA_ARCHITECTURES for TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
    endif()
    
    # Use the cmake config provided by torch
    find_package(Torch REQUIRED PATHS "${Pytorch_ROOT}"
                 NO_DEFAULT_PATH)

    if(BUILD_CUDA_MODULE)
        # Note: older versions of PyTorch have hard-coded cuda library paths, see:
        # https://github.com/pytorch/pytorch/issues/15476.
        # This issue has been addressed but we observed for the conda packages for
        # PyTorch 1.2.0 and 1.4.0 that there are still hardcoded paths in
        #  ${TORCH_ROOT}/share/cmake/Caffe2/Caffe2Targets.cmake
        # Try to fix those here
        find_package(CUDAToolkit REQUIRED)
        get_target_property( iface_link_libs torch INTERFACE_LINK_LIBRARIES )
        string( REPLACE "/usr/local/cuda" "${CUDAToolkit_LIBRARY_ROOT}" iface_link_libs "${iface_link_libs}" )
        set_target_properties( torch PROPERTIES INTERFACE_LINK_LIBRARIES "${iface_link_libs}" )
        if( TARGET torch_cuda )
            get_target_property( iface_link_libs torch_cuda INTERFACE_LINK_LIBRARIES )
            string( REPLACE "/usr/local/cuda" "${CUDAToolkit_LIBRARY_ROOT}" iface_link_libs "${iface_link_libs}" )
            set_target_properties( torch_cuda PROPERTIES INTERFACE_LINK_LIBRARIES "${iface_link_libs}" )
        endif()
        # if successful everything works :)
        # if unsuccessful CMake will complain that there are no rules to make the targets with the hardcoded paths

        # remove flags that nvcc does not understand
        get_target_property( iface_compile_options torch INTERFACE_COMPILE_OPTIONS )
        set_target_properties( torch PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
        set_target_properties( torch_cuda PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
        set_target_properties( torch_cpu PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
    endif()

    # If MKL is installed in the system level (e.g. for oneAPI Toolkit),
    # caffe2::mkl and caffe2::mkldnn will be added to torch_cpu's
    # INTERFACE_LINK_LIBRARIES. However, Open3D already comes with MKL linkage
    # and we're not using MKLDNN.
    get_target_property(torch_cpu_INTERFACE_LINK_LIBRARIES torch_cpu
                        INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM torch_cpu_INTERFACE_LINK_LIBRARIES caffe2::mkl)
    list(REMOVE_ITEM torch_cpu_INTERFACE_LINK_LIBRARIES caffe2::mkldnn)
    set_target_properties(torch_cpu PROPERTIES INTERFACE_LINK_LIBRARIES
                          "${torch_cpu_INTERFACE_LINK_LIBRARIES}")
endif()

message(STATUS "PyTorch         version: ${Pytorch_VERSION}")
message(STATUS "               root dir: ${Pytorch_ROOT}")
message(STATUS "          compile flags: ${TORCH_CXX_FLAGS}")
if (UNIX AND NOT APPLE)
    message(STATUS "          use cxx11 abi: ${Pytorch_CXX11_ABI}")
endif()
foreach(idir ${TORCH_INCLUDE_DIRS})
    message(STATUS "           include dirs: ${idir}")
endforeach(idir)
foreach(lib ${TORCH_LIBRARIES})
    message(STATUS "              libraries: ${lib}")
endforeach(lib)

# Check if the c++11 ABI is compatible on Linux
if(UNIX AND NOT APPLE)
    if((Pytorch_CXX11_ABI AND (NOT GLIBCXX_USE_CXX11_ABI)) OR
       (NOT Pytorch_CXX11_ABI AND GLIBCXX_USE_CXX11_ABI))
        if(Pytorch_CXX11_ABI)
            set(NEEDED_ABI_FLAG "ON")
        else()
            set(NEEDED_ABI_FLAG "OFF")
        endif()
        message(FATAL_ERROR "PyTorch and Open3D ABI mismatch: ${Pytorch_CXX11_ABI} != ${GLIBCXX_USE_CXX11_ABI}.\n"
                            "Please use -DGLIBCXX_USE_CXX11_ABI=${NEEDED_ABI_FLAG} "
                            "in the cmake config command to change the Open3D ABI.")
    else()
        message(STATUS "PyTorch matches Open3D ABI: ${Pytorch_CXX11_ABI} == ${GLIBCXX_USE_CXX11_ABI}")
    endif()
endif()

message(STATUS "Pytorch_VERSION: ${Pytorch_VERSION}, CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
if (BUILD_PYTORCH_OPS AND BUILD_CUDA_MODULE AND CUDAToolkit_VERSION
        VERSION_GREATER_EQUAL "11.0" AND Pytorch_VERSION VERSION_LESS
        "1.9")
    message(WARNING
        "--------------------------------------------------------------------------------\n"
        "                                                                                \n"
        " You are compiling PyTorch ops with CUDA 11 with PyTorch version < 1.9. This    \n"
        " configuration may have stability issues. See                                   \n"
        " https://github.com/isl-org/Open3D/issues/3324 and                              \n"
        " https://github.com/pytorch/pytorch/issues/52663 for more information on this   \n"
        " problem.                                                                       \n"
        "                                                                                \n"
        " We recommend to compile PyTorch from source with compile flags                 \n"
        "   '-Xcompiler -fno-gnu-unique'                                                 \n"
        "                                                                                \n"
        " or use the PyTorch wheels at                                                   \n"
        "   https://github.com/isl-org/open3d_downloads/releases/tag/torch1.8.2          \n"
        "                                                                                \n"
        "--------------------------------------------------------------------------------\n"
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pytorch DEFAULT_MSG Pytorch_VERSION
                                  Pytorch_ROOT)
