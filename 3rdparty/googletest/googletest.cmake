include(FetchContent)

# Open3D compiles gtest sources into Open3D::3rdparty_googletest; do not install
# the FetchContent targets (shared libgmock.so may be missing / unused).
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    ext_googletest
    PREFIX googletest
    URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
    URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/googletest"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(ext_googletest)
FetchContent_GetProperties(ext_googletest SOURCE_DIR GOOGLETEST_SOURCE_DIR)

# Suppress -Wcharacter-conversion error in googletest when using IntelLLVM/ICX.
# ICX's Clang front-end treats the implicit char16_t->char32_t conversion in
# gtest-printers.h(524) as a hard error. Suppress it on FetchContent targets.
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    foreach(_gt_tgt gtest gtest_main gmock gmock_main)
        if(TARGET ${_gt_tgt})
            target_compile_options(${_gt_tgt} PRIVATE -Wno-character-conversion)
        endif()
    endforeach()
endif()
