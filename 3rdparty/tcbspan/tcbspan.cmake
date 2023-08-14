include(ExternalProject)

set(TCB_SPAN_LIB_NAME tcbspan)

ExternalProject_Add(
    ext_tcbspan
    PREFIX tcbspan
    GIT_REPOSITORY https://github.com/tcbrindle/span.git
    GIT_TAG 836dc6a0efd9849cb194e88e4aa2387436bb079b
    GIT_SHALLOW TRUE
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/tcbspan"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tcbspan/CMakeLists.txt <SOURCE_DIR>
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)
ExternalProject_Get_Property(ext_tcbspan DOWNLOAD_DIR)
message(WARNING "TCBSpan source dir ${DOWNLOAD_DIR}")

ExternalProject_Get_Property(ext_tcbspan INSTALL_DIR)
set(TCB_SPAN_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(STDGPU_LIBRARIES tcb::span)
