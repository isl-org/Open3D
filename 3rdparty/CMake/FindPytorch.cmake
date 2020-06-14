# Find the Pytorch root and use the provided cmake module
#
# The following variables will be set
#   Pytorch_FOUND
#   Pytorch_VERSION
#   Pytorch_ROOT
#   Pytorch_DEFINITIONS
#
# This script will call find_package( Torch ) which will define
#   TORCH_FOUND
#   TORCH_INCLUDE_DIRS
#   TORCH_LIBRARIES
#   TORCH_CXX_FLAGS
#
# and import the target 'torch'
#
if (NOT Pytorch_FOUND)
    # searching for pytorch requires the python executable
    find_package( PythonExecutable REQUIRED )

    message( STATUS "Getting Pytorch properties ..." )

    execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "import torch; print(torch.__version__, end='')"
            OUTPUT_VARIABLE Pytorch_VERSION )
    execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "import os; import torch; print(os.path.dirname(torch.__file__), end='')"
            OUTPUT_VARIABLE Pytorch_ROOT )

    # use the cmake config provided by torch
    find_package( Torch REQUIRED
                  PATHS "${Pytorch_ROOT}/share/cmake/Torch"
                  NO_DEFAULT_PATH )

    # There are hardcoded paths in ${TORCH_ROOT}/share/cmake/Caffe2/Caffe2Targets.cmake (observed for Pytorch 1.2.0)
    # Try to fix those here
    get_target_property( iface_link_libs torch INTERFACE_LINK_LIBRARIES )
    string( REPLACE "/usr/local/cuda/lib64" "${CUDA_TOOLKIT_ROOT_DIR}" iface_link_libs "${iface_link_libs}" )
    set_target_properties( torch PROPERTIES INTERFACE_LINK_LIBRARIES "${iface_link_libs}" )
    # if successful everything works :)
    # if unsuccessful CMake will complain that there are no rules to make the targets with the hardcoded paths
endif()

message( STATUS "Pytorch         version: ${Pytorch_VERSION}" )
message( STATUS "               root dir: ${Pytorch_ROOT}" )
message( STATUS "          compile flags: ${TORCH_CXX_FLAGS}" )
foreach( idir ${TORCH_INCLUDE_DIRS} )
message( STATUS "           include dirs: ${idir}" )
endforeach( idir )
foreach( lib ${TORCH_LIBRARIES} )
message( STATUS "              libraries: ${lib}" )
endforeach( lib )

# check if the c++11 ABI is compatible
if( ${TORCH_CXX_FLAGS} MATCHES "_GLIBCXX_USE_CXX11_ABI=([01])" )
        if( (GLIBCXX_USE_CXX11_ABI AND NOT CMAKE_MATCH_1) OR (NOT GLIBCXX_USE_CXX11_ABI AND CMAKE_MATCH_1) )
                message( FATAL_ERROR "_GLIBCXX_USE_CXX11_ABI mismatch: Open3D ${GLIBCXX_USE_CXX11_ABI} vs Torch ${CMAKE_MATCH_1}")
        endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pytorch DEFAULT_MSG Pytorch_VERSION Pytorch_ROOT )

