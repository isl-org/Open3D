include(ExternalProject)

if (WIN32)
    set(LIBDIR "lib")
else()
    include(GNUInstallDirs)
    set(LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()

# Set compiler flags
# if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
# endif()

# Set WITH_SIMD
# include(CheckLanguage)
# check_language(ASM_NASM)
# if (CMAKE_ASM_NASM_COMPILER)
#     if (APPLE)
#         # macOS might have /usr/bin/nasm but it cannot be used
#         # https://stackoverflow.com/q/53974320
#         # To fix this, run `brew install nasm`
#         execute_process(COMMAND nasm --version RESULT_VARIABLE return_code)
#         if("${return_code}" STREQUAL "0")
#             enable_language(ASM_NASM)
#             option(WITH_SIMD "" ON)
#         else()
#             message(STATUS "nasm found but can't be used, run `brew install nasm`")
#             option(WITH_SIMD "" OFF)
#         endif()
#     else()
#         enable_language(ASM_NASM)
#         option(WITH_SIMD "" ON)
#     endif()
# else()
#     option(WITH_SIMD "" OFF)
# endif()
# if (WITH_SIMD)
#     message(STATUS "NASM assembler enabled")
# else()
#     message(STATUS "NASM assembler not found - libjpeg-turbo performance may suffer")
# endif()

# if (STATIC_WINDOWS_RUNTIME)
#     set(WITH_CRT_DLL OFF)
# else()
#     set(WITH_CRT_DLL ON)
# endif()
# message(STATUS "libturbojpeg: WITH_CRT_DLL=${WITH_CRT_DLL}")

ExternalProject_Add(
    libfaiss
    PREFIX faiss
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/faiss/faiss
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND -L/opt/anaconda3/pkgs/mkl-2019.3-199/lib ${CMAKE_CURRENT_SOURCE_DIR}/faiss/faiss/configure --without-cuda --prefix ${3RDPARTY_INSTALL_PREFIX}
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    # CMAKE_GENERATOR ${CMAKE_GENERATOR}
    # CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    # CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    # CMAKE_ARGS
    #     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    #     -DENABLE_STATIC=ON
    #     -DENABLE_SHARED=OFF
    #     -DCMAKE_INSTALL_PREFIX=${3RDPARTY_INSTALL_PREFIX}
    #     -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

add_library(faiss INTERFACE)
add_dependencies(faiss libfaiss)

# If MSVC, the OUTPUT_NAME was set to turbojpeg-static
if(MSVC)
    set(lib_name "faiss-static")
else()
    set(lib_name "faiss")
endif()

# For linking with Open3D's after installation
set(FAISS_LIBRARIES ${lib_name})

set(faiss_LIB_FILES
    ${3RDPARTY_INSTALL_PREFIX}/${LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
)

target_include_directories(faiss SYSTEM INTERFACE
    ${3RDPARTY_INSTALL_PREFIX}/include
)
target_link_libraries(faiss INTERFACE
    ${faiss_LIB_FILES}
)

if (NOT BUILD_SHARED_LIBS)
    install(FILES ${faiss_LIB_FILES}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

add_dependencies(build_all_3rd_party_libs faiss)


