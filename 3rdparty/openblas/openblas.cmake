include(ExternalProject)

set(OPENBLAS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/openblas)

if(WIN32)
    ExternalProject_Add(
        ext_openblas
        PREFIX openblas
        GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
        GIT_TAG v0.3.10
        CMAKE_ARGS
            -DMSVC_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}
            -DCMAKE_INSTALL_PREFIX=${OPENBLAS_INSTALL_PREFIX}
    )
    set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_INSTALL_PREFIX}/include/openblas/")
    set(OPENBLAS_LIB_DIR "${OPENBLAS_INSTALL_PREFIX}/lib")
    set(OPENBLAS_LIBRARIES openblas)  # Extends to openblas.lib automatically.
else()
    set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_INSTALL_PREFIX}/include/") # The "/"" is critical, see import_3rdparty_library.
    set(OPENBLAS_LIB_DIR "${OPENBLAS_INSTALL_PREFIX}/lib")
    set(OPENBLAS_LIBRARIES openblas)  # Extends to libopenblas.a automatically.
    ExternalProject_Add(
        ext_openblas
        PREFIX openblas
        GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
        GIT_TAG v0.3.10
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND $(MAKE) TARGET=NEHALEM NO_SHARED=1 LIBNAME=CUSTOM_LIB_NAME
        BUILD_IN_SOURCE True
        INSTALL_COMMAND $(MAKE) install PREFIX=${OPENBLAS_INSTALL_PREFIX} NO_SHARED=1 LIBNAME=CUSTOM_LIB_NAME
        COMMAND ${CMAKE_COMMAND} -E rename ${OPENBLAS_LIB_DIR}/CUSTOM_LIB_NAME ${OPENBLAS_LIB_DIR}/libopenblas.a
    )
endif()

message(STATUS "OPENBLAS_INCLUDE_DIR: ${OPENBLAS_INCLUDE_DIR}")
message(STATUS "OPENBLAS_LIB_DIR ${OPENBLAS_LIB_DIR}")
message(STATUS "OPENBLAS_LIBRARIES: ${OPENBLAS_LIBRARIES}")
