# open3d_set_global_properties(target)
#
# Sets important project-related properties to <target>.
function(open3d_set_global_properties target)
    # Tell CMake we want a compiler that supports C++14 features
    target_compile_features(${target} PUBLIC cxx_std_17)

    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
        $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}>
    )

    target_compile_definitions(${target} PRIVATE UNIX)

    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")

    # Require 64-bit indexing in vectorized code
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:ISPC>:--addressing=64>)

    # Turn off fast math for IntelLLVM DPC++ compiler
    # Fast math is turned off for clang by default even for -O3.
    # We may make this optional and tune unit tests floating point precisions.
    target_compile_options(${target} PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<COMPILE_LANGUAGE:ISPC>>>:-fno-fast-math>)

    if (GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    else()
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    endif()

    # TBB static version is used
    # See: https://github.com/wjakob/tbb/commit/615d690c165d68088c32b6756c430261b309b79c
    target_compile_definitions(${target} PRIVATE __TBB_LIB_NAME=tbb_static)

endfunction()
