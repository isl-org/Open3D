include(ExternalProject)

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

set(FAISS_ROOT "${CMAKE_CURRENT_BINARY_DIR}/libfaiss_install")

ExternalProject_Add(
    ext_faiss
    PREFIX faiss
    GIT_REPOSITORY https://github.com/junha-l/faiss.git 
    GIT_TAG 954ada2cc1106bd8f20c0f99bff615e36c0053b1
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${FAISS_ROOT}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_CUDA_FLAGS=${CUDA_GENCODES}
        -DCUDAToolkit_ROOT=${CUDA_TOOLKIT_ROOT_DIR}
        -DFAISS_ENABLE_GPU=${BUILD_CUDA_MODULE}
        -DFAISS_ENABLE_PYTHON=OFF
        -DFAISS_USE_SYSTEM_BLAS=OFF
        -DFAISS_BLAS_TARGET=${FAISS_BLAS_TARGET}
        -DBUILD_TESTING=OFF
)

if(MSVC)
    set(lib_name "faiss-static")
else()
    set(lib_name "faiss")
endif()

set(FAISS_LIBRARIES ${lib_name})
set(FAISS_INCLUDE_DIR "${FAISS_ROOT}/include/")
set(FAISS_LIB_DIR "${FAISS_ROOT}/lib/")
