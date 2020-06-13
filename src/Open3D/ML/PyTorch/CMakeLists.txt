message( STATUS "Building Pytorch ops" )

if (BUILD_CUDA_MODULE)
    set( CUDA_ENABLED YES )
    message(STATUS "Building Pytorch ops with CUDA")
endif ()


find_package( Pytorch REQUIRED )
find_package( TBB REQUIRED )
find_package( nanoflann REQUIRED )
find_package( parallelstl REQUIRED )
if ( CUDA_ENABLED )
    find_package( CUB REQUIRED )
    #find_package( CUTLASS REQUIRED )
endif ()


if ( CUDA_ENABLED )
        file(GLOB_RECURSE open3d_torch_ops_SOURCES
                "*.cpp"
                "*.cu"
                "../*/Detail/*.cc"
                "../*/Detail/*.cpp"
                "../*/Detail/*.cu" )
else ()
    file(GLOB_RECURSE open3d_torch_ops_SOURCES
            "*.cpp"
            "../*/Detail/*.cc"
            "../*/Detail/*.cpp" )
endif ()
list(APPEND open3d_torch_ops_SOURCES
        "../../Utility/Console.cpp"
        "../../Utility/Helper.cpp"
        )

message( STATUS ${open3d_torch_ops_SOURCES})
add_library( open3d_torch_ops SHARED ${open3d_torch_ops_SOURCES} )

set_target_properties( open3d_torch_ops PROPERTIES PREFIX "" ) # do not add 'lib' prefix
set_target_properties( open3d_torch_ops PROPERTIES DEBUG_POSTFIX "_debug" )
# Use flags from Torch . This is usually just a definition for the GLIBC ABI.
set_target_properties( open3d_torch_ops PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS}" )
if( CUDA_ENABLED )
        set_target_properties( open3d_torch_ops PROPERTIES COMPILE_DEFINITIONS "CUDA_ENABLED" )
endif()

# enable c++11
set_target_properties( open3d_torch_ops PROPERTIES
                       CXX_STANDARD 11
                       CXX_STANDARD_REQUIRED ON
                       CXX_EXTENSIONS OFF
                       CUDA_STANDARD 11
                       )

target_include_directories( open3d_torch_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${TBB_INCLUDE_DIR}
    ${nanoflann_INCLUDE_DIR}
    ${parallelstl_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    )
if ( CUDA_ENABLED )
    target_include_directories( open3d_torch_ops PRIVATE
        ${CUB_INCLUDE_DIR}
        #${CUTLASS_INCLUDE_DIR}
        )
endif ()

target_link_libraries( open3d_torch_ops PRIVATE
    ${TBB_LIBRARY}
    ${TBB_MALLOC_LIBRARY}
    ${TORCH_LIBRARIES}
    ${EIGEN3_TARGET}
    ${FMT_TARGET}
    )

