message( STATUS "Building Tensorflow ops" )

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if ("CUDA" IN_LIST languages)
    set( CUDA_ENABLED YES )
    message(STATUS "Building Tensorflow ops with CUDA")
endif ()


find_package( Tensorflow REQUIRED )
find_package( TBB REQUIRED )
#find_package( nanoflann REQUIRED )
#find_package( parallelstl REQUIRED )
#if ( CUDA_ENABLED )
    #find_package( CUB REQUIRED )
    #find_package( CUTLASS REQUIRED )
#endif ()


if ( CUDA_ENABLED )
        file(GLOB_RECURSE open3d_tf_ops_SOURCES 
                "*.cc" 
                "*.cpp" 
                "*.cu" 
                "../Open3D/ML/*/Detail/*.cc" 
                "../Open3D/ML/*/Detail/*.cpp" 
                "../Open3D/ML/*/Detail/*.cu" )
else ()
    file(GLOB_RECURSE open3d_tf_ops_SOURCES 
            "*.cc" 
            "*.cpp" 
            "../Open3D/ML/*/Detail/*.cc" 
            "../Open3D/ML/*/Detail/*.cpp" )
endif ()

add_library( open3d_tf_ops SHARED ${open3d_tf_ops_SOURCES} )
open3d_set_global_properties(open3d_tf_ops)
open3d_set_global_properties(open3d_tf_ops)

set_target_properties( open3d_tf_ops PROPERTIES PREFIX "" ) # do not add 'lib' prefix
set_target_properties( open3d_tf_ops PROPERTIES DEBUG_POSTFIX "_debug" )
# Use definitions from tensorflow. This is usually just a definition for the GLIBC ABI.
foreach( definition IN LISTS Tensorflow_DEFINITIONS )
	set_target_properties( open3d_tf_ops PROPERTIES COMPILE_DEFINITIONS "${definition}" )
endforeach(definition)

target_include_directories( open3d_tf_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    #${nanoflann_INCLUDE_DIR}
    #${parallelstl_INCLUDE_DIR}
    )
# silence warnings caused by tensorflow's Eigen
# Travis has a different version of protobuf in /usr/local/include
# TBB is in /usr/local/include, so it needs to be after Tensorflow
target_include_directories( open3d_tf_ops SYSTEM PRIVATE
    ${Tensorflow_INCLUDE_DIR}
    ${TBB_INCLUDE_DIR}
    )
#if ( CUDA_ENABLED )
    #target_include_directories( open3d_tf_ops PRIVATE
        #${CUB_INCLUDE_DIR}
        #${CUTLASS_INCLUDE_DIR}
        #)
#endif ()

target_link_libraries( open3d_tf_ops PRIVATE
    ${TBB_LIBRARY}
    ${TBB_MALLOC_LIBRARY}
    ${Tensorflow_FRAMEWORK_LIB}
)
