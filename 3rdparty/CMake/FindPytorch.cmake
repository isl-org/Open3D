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

if(NOT Pytorch_FOUND)
    # Searching for pytorch requires the python executable
    find_package(PythonExecutable REQUIRED)

    message(STATUS "Getting PyTorch properties ...")

    # Get Pytorch_VERSION
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
                "import torch; print(torch.__version__, end='')"
        OUTPUT_VARIABLE Pytorch_VERSION)

    # Get Pytorch_ROOT
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} "-c"
            "import os; import torch; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE Pytorch_ROOT)

    # Use the cmake config provided by torch
    find_package(Torch REQUIRED PATHS "${Pytorch_ROOT}/share/cmake/Torch"
                 NO_DEFAULT_PATH)

    # Note: older versions of PyTorch have hard-coded cuda library paths, see:
    # https://github.com/pytorch/pytorch/issues/15476.
    # This issue has been addressed but we observed for the conda packages for
    # PyTorch 1.2.0 and 1.4.0 that there are still hardcoded paths in
    #  ${TORCH_ROOT}/share/cmake/Caffe2/Caffe2Targets.cmake
    # Try to fix those here
    get_target_property( iface_link_libs torch INTERFACE_LINK_LIBRARIES )
    string( REPLACE "/usr/local/cuda" "${CUDA_TOOLKIT_ROOT_DIR}" iface_link_libs "${iface_link_libs}" )
    set_target_properties( torch PROPERTIES INTERFACE_LINK_LIBRARIES "${iface_link_libs}" )
    if( TARGET torch_cuda )
        get_target_property( iface_link_libs torch_cuda INTERFACE_LINK_LIBRARIES )
        string( REPLACE "/usr/local/cuda" "${CUDA_TOOLKIT_ROOT_DIR}" iface_link_libs "${iface_link_libs}" )
        set_target_properties( torch_cuda PROPERTIES INTERFACE_LINK_LIBRARIES "${iface_link_libs}" )
    endif()
    # if successful everything works :)
    # if unsuccessful CMake will complain that there are no rules to make the targets with the hardcoded paths

    if( BUILD_CUDA_MODULE )
        # remove flags that nvcc does not understand
        get_target_property( iface_compile_options torch INTERFACE_COMPILE_OPTIONS )
        set_target_properties( torch PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
        set_target_properties( torch_cuda PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
        set_target_properties( torch_cpu PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
    endif()

    # Get Pytorch_CXX11_ABI: True/False
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} "-c"
            "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI, end='')"
        OUTPUT_VARIABLE Pytorch_CXX11_ABI
    )
endif()

message(STATUS "PyTorch         version: ${Pytorch_VERSION}")
message(STATUS "               root dir: ${Pytorch_ROOT}")
message(STATUS "          compile flags: ${TORCH_CXX_FLAGS}")
message(STATUS "          use cxx11 abi: ${Pytorch_CXX11_ABI}")
foreach(idir ${TORCH_INCLUDE_DIRS})
    message(STATUS "           include dirs: ${idir}")
endforeach(idir)
foreach(lib ${TORCH_LIBRARIES})
    message(STATUS "              libraries: ${lib}")
endforeach(lib)

# Check if the c++11 ABI is compatible
if((Pytorch_CXX11_ABI AND (NOT GLIBCXX_USE_CXX11_ABI)) OR
   (NOT Pytorch_CXX11_ABI AND GLIBCXX_USE_CXX11_ABI))
    message(FATAL_ERROR "PyTorch and Open3D ABI mismatch: ${Pytorch_CXX11_ABI} != ${GLIBCXX_USE_CXX11_ABI}")
else()
    message(STATUS "PyTorch matches Open3D ABI: ${Pytorch_CXX11_ABI} == ${GLIBCXX_USE_CXX11_ABI}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pytorch DEFAULT_MSG Pytorch_VERSION
                                  Pytorch_ROOT)
