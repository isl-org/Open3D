include(ExternalProject)

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

set(FAISS_ROOT "${CMAKE_CURRENT_BINARY_DIR}/libfaiss_install")
if (BUILD_CUDA_MODULE) 
    message(STATUS "${CUDA_TOOLKIT_ROOT_DIR}")
    ExternalProject_Add(
        ext_faiss
        PREFIX faiss
        SOURCE_DIR ${Open3D_3RDPARTY_DIR}/faiss/faiss
        CONFIGURE_COMMAND ${Open3D_3RDPARTY_DIR}/faiss/faiss/configure --with-cuda=${CUDA_TOOLKIT_ROOT_DIR} --prefix=${FAISS_ROOT}
        BUILD_COMMAND ${MAKE}
        BUILD_IN_SOURCE 1
    )
else()
    ExternalProject_Add(
        ext_faiss
        PREFIX faiss
        SOURCE_DIR ${Open3D_3RDPARTY_DIR}/faiss/faiss
        CONFIGURE_COMMAND ${Open3D_3RDPARTY_DIR}/faiss/faiss/configure --without-cuda --prefix=${FAISS_ROOT}
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
