include(ExternalProject)

if(WIN32)
    set(OPENBLAS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/openblas)
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
    ExternalProject_Add(
        ext_openblas
        PREFIX openblas
        GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
        GIT_TAG v0.3.10
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E rename "libopenblas_nehalemp-r0.3.10.a" "libopenblas.a"
        # OpenBLAS builds in source directory.
        BUILD_IN_SOURCE True
        # Avoids libopenblas.so, only use libopenblas.a.
        BUILD_COMMAND make TARGET=NEHALEM NO_SHARED=1
    )
    ExternalProject_Get_property(ext_openblas SOURCE_DIR)
    set(OPENBLAS_INCLUDE_DIR "${SOURCE_DIR}/") # The "/"" is critical, see import_3rdparty_library.
    set(OPENBLAS_LIB_DIR "${SOURCE_DIR}")
    set(OPENBLAS_LIBRARIES openblas)  # Extends to libopenblas.a automatically.
endif()

message(STATUS "OPENBLAS_INCLUDE_DIR: ${OPENBLAS_INCLUDE_DIR}")
message(STATUS "OPENBLAS_LIB_DIR ${OPENBLAS_LIB_DIR}")
message(STATUS "OPENBLAS_LIBRARIES: ${OPENBLAS_LIBRARIES}")
