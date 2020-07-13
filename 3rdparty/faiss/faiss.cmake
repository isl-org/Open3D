include(ExternalProject)

find_package(BLAS REQUIRED)

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

if (BUILD_CUDA_MODULE) 
    ExternalProject_Add(
        ext_faiss
        PREFIX faiss
        SOURCE_DIR ${Open3D_3RDPARTY_DIR}/faiss/faiss
        CONFIGURE_COMMAND ${Open3D_3RDPARTY_DIR}/faiss/faiss/configure --with-cuda=${CUDA_TOOLKIT_ROOT_DIR}  --prefix=${CMAKE_CURRENT_BINARY_DIR}/libfaiss-install
        BUILD_COMMAND ${MAKE}
        BUILD_IN_SOURCE 1
    )
else()
    ExternalProject_Add(
        ext_faiss
        PREFIX faiss
        SOURCE_DIR ${Open3D_3RDPARTY_DIR}/faiss/faiss
        CONFIGURE_COMMAND ${Open3D_3RDPARTY_DIR}/faiss/faiss/configure --without-cuda  --prefix=${CMAKE_CURRENT_BINARY_DIR}/libfaiss-install
        BUILD_COMMAND ${MAKE}
        BUILD_IN_SOURCE 1
    )
endif()

if(MSVC)
    set(lib_name "faiss-static")
else()
    set(lib_name "faiss")
endif()

set(FAISS_LIBRARIES ${lib_name})
