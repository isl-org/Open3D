include(ExternalProject)

# MKL_INSTALL_PREFIX contains MKL-TBB or BLAS static library.
# Faiss depends on MKL-TBB or BLAS. We put them in the same directory so that
# FAISS_LIBRARIES can refer to libraries in the same directory.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)

ExternalProject_Add(
    ext_faiss
    PREFIX faiss
    URL https://github.com/isl-org/faiss/archive/0493ff79e50dc4c5a55ebf0af4b7984d86f8227d.tar.gz # open3d_patch branch
    URL_HASH SHA256=6550aa32ea28484ec774228b5cc7555c58304dd971bb5e5601999c351f20b9bd
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/faiss"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${MKL_INSTALL_PREFIX}
        -DCMAKE_CUDA_FLAGS=${CUDA_GENCODES}
        -DCUDAToolkit_ROOT=${CUDAToolkit_LIBRARY_ROOT}
        -DFAISS_ENABLE_GPU=${BUILD_CUDA_MODULE}
        -DFAISS_ENABLE_PYTHON=OFF
        -DFAISS_USE_SYSTEM_BLAS=OFF
        -DBUILD_TESTING=OFF
        ${ExternalProject_CMAKE_ARGS_hidden}
    CMAKE_CACHE_ARGS    # Lists must be passed via CMAKE_CACHE_ARGS
        -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
    BUILD_BYPRODUCTS
        ${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}faiss${CMAKE_STATIC_LIBRARY_SUFFIX}
)

set(FAISS_LIBRARIES faiss)
set(FAISS_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/include/")
set(FAISS_LIB_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_LIB_DIR}")  # Must have no trailing "/".
