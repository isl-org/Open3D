include(ExternalProject)

# MKL_INSTALL_PREFIX contains MKL-TBB or BLAS static library.
# Faiss depends on MKL-TBB or BLAS. We put them in the same directory so that
# FAISS_LIBRARIES can refer to libraries in the same directory.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)

ExternalProject_Add(
    ext_faiss
    PREFIX faiss
    GIT_REPOSITORY https://github.com/junha-l/faiss.git
    GIT_TAG 954ada2cc1106bd8f20c0f99bff615e36c0053b1
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${MKL_INSTALL_PREFIX}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_CUDA_FLAGS=${CUDA_GENCODES}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCUDAToolkit_ROOT=${CUDA_TOOLKIT_ROOT_DIR}
        -DFAISS_ENABLE_GPU=${BUILD_CUDA_MODULE}
        -DFAISS_ENABLE_PYTHON=OFF
        -DFAISS_USE_SYSTEM_BLAS=OFF
        -DBUILD_TESTING=OFF
)

set(FAISS_LIBRARIES faiss)
set(FAISS_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/include/")
set(FAISS_LIB_DIR "${MKL_INSTALL_PREFIX}/lib")  # Must have no trailing "/".
