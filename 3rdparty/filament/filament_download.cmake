include(FetchContent)
include(ExternalProject)

set(filament_LIBRARIES filameshio filament filaflat filabridge geometry backend bluegl bluevk ibl image ktxreader meshoptimizer smol-v utils vkshaders)

if (FILAMENT_PRECOMPILED_ROOT)
    if (EXISTS "${FILAMENT_PRECOMPILED_ROOT}")
        set(FILAMENT_ROOT "${FILAMENT_PRECOMPILED_ROOT}")
    else()
        message(FATAL_ERROR "Filament binaries not found in ${FILAMENT_PRECOMPILED_ROOT}")
    endif()
else()
    # Locate byproducts
    set(lib_dir lib)
    # Setup download links
    if(WIN32)
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.54.0/filament-v1.54.0-windows.tgz)
        set(FILAMENT_SHA256 370b85dbaf1a3be26a5a80f60c912f11887748ddd1c42796a83fe989f5805f7b)
        if (STATIC_WINDOWS_RUNTIME)
            string(APPEND lib_dir /x86_64/mt)
        else()
            string(APPEND lib_dir /x86_64/md)
        endif()
    elseif(APPLE)
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.54.0/filament-v1.54.0-mac.tgz)
        set(FILAMENT_SHA256 9b71642bd697075110579ccb55a2e8f319b05bbd89613c72567745534936186e)
    else()      # Linux: Check glibc version and use open3d filament binary if new (Ubuntu 20.04 and similar)
        execute_process(COMMAND ldd --version OUTPUT_VARIABLE ldd_version)
        string(REGEX MATCH "([0-9]+\.)+[0-9]+" glibc_version ${ldd_version})
        if(${glibc_version} VERSION_LESS "2.33")
            set(FILAMENT_URL
                    https://github.com/isl-org/open3d_downloads/releases/download/filament/filament-v1.49.1-ubuntu20.04.tgz)
            set(FILAMENT_SHA256 f4ba020f0ca63540e2f86b36d1728a1ea063ddd5eb55b0ba6fc621ee815a60a7)
            message(STATUS "GLIBC version ${glibc_version} found: Using "
                    "Open3D built Filament binary for Ubuntu 20.04.")
        else()
            set(FILAMENT_URL
                    https://github.com/google/filament/releases/download/v1.54.0/filament-v1.54.0-linux.tgz)
            set(FILAMENT_SHA256 f07fbe8fcb6422a682f429d95fa2e097c538d0d900c62f0a835f595ab3909e8e)
            message(STATUS "GLIBC version ${glibc_version} found: Using "
                    "Google Filament binary.")
        endif()
    endif()

    set(lib_byproducts ${filament_LIBRARIES})
    list(TRANSFORM lib_byproducts PREPEND <SOURCE_DIR>/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
    list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
    message(STATUS "Filament byproducts: ${lib_byproducts}")

    if(WIN32)
        set(lib_byproducts_debug ${filament_LIBRARIES})
        list(TRANSFORM lib_byproducts_debug PREPEND <SOURCE_DIR>/${lib_dir}d/${CMAKE_STATIC_LIBRARY_PREFIX})
        list(TRANSFORM lib_byproducts_debug APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
        list(APPEND lib_byproducts ${lib_byproducts_debug})
    endif()

    # ExternalProject_Add happens at build time.
    ExternalProject_Add(
            ext_filament
            PREFIX filament
            URL ${FILAMENT_URL}
            URL_HASH SHA256=${FILAMENT_SHA256}
            DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/filament"
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_IN_SOURCE ON
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS ${lib_byproducts}
    )
    ExternalProject_Get_Property(ext_filament SOURCE_DIR)
    message(STATUS "Filament source dir is ${SOURCE_DIR}")
    set(FILAMENT_ROOT ${SOURCE_DIR})
    
    # On Linux, build matc from source to avoid GLIBC version issues
    if(UNIX AND NOT APPLE)
        message(STATUS "Building matc from source to ensure GLIBC compatibility")
        
        # Determine compiler for matc (Filament requires Clang >= 7)
        set(MATC_C_COMPILER "${CMAKE_C_COMPILER}")
        set(MATC_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
        
        if(NOT MSVC AND NOT (CMAKE_C_COMPILER_ID MATCHES ".*Clang" AND
            CMAKE_CXX_COMPILER_ID MATCHES ".*Clang"
            AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7))
            # Try to find a suitable Clang version
            find_program(CLANG_VERSIONED_CXX NAMES
                         clang++-19 clang++-18 clang++-17 clang++-16 clang++-15
                         clang++-14 clang++-13 clang++-12 clang++-11 clang++-10
                         clang++-9 clang++-8 clang++-7
            )
            if(CLANG_VERSIONED_CXX)
                get_filename_component(CLANG_VERSIONED_CXX_NAME "${CLANG_VERSIONED_CXX}" NAME_WE)
                string(REPLACE "++" "" CLANG_VERSIONED_CC_NAME "${CLANG_VERSIONED_CXX_NAME}")
                get_filename_component(CLANG_VERSIONED_CXX_DIR "${CLANG_VERSIONED_CXX}" DIRECTORY)
                find_program(CLANG_VERSIONED_CC_FULL 
                             NAMES ${CLANG_VERSIONED_CC_NAME}
                             PATHS ${CLANG_VERSIONED_CXX_DIR}
                             NO_DEFAULT_PATH
                )
                if(CLANG_VERSIONED_CC_FULL)
                    set(MATC_C_COMPILER "${CLANG_VERSIONED_CC_FULL}")
                    set(MATC_CXX_COMPILER "${CLANG_VERSIONED_CXX}")
                    message(STATUS "Found Clang for matc: ${MATC_CXX_COMPILER}")
                endif()
            endif()
            
            # Fallback to default clang++
            if(NOT MATC_CXX_COMPILER OR MATC_CXX_COMPILER STREQUAL CMAKE_CXX_COMPILER)
                find_program(CLANG_DEFAULT_CXX NAMES clang++)
                if(CLANG_DEFAULT_CXX)
                    execute_process(COMMAND ${CLANG_DEFAULT_CXX} --version OUTPUT_VARIABLE clang_version ERROR_QUIET)
                    if(clang_version MATCHES "clang version ([0-9]+)")
                        if (CMAKE_MATCH_1 GREATER_EQUAL 7)
                            get_filename_component(CLANG_DEFAULT_CXX_DIR "${CLANG_DEFAULT_CXX}" DIRECTORY)
                            find_program(CLANG_DEFAULT_CC_FULL NAMES clang PATHS ${CLANG_DEFAULT_CXX_DIR} NO_DEFAULT_PATH)
                            if(CLANG_DEFAULT_CC_FULL)
                                set(MATC_C_COMPILER "${CLANG_DEFAULT_CC_FULL}")
                                set(MATC_CXX_COMPILER "${CLANG_DEFAULT_CXX}")
                                message(STATUS "Found default Clang for matc: ${MATC_CXX_COMPILER}")
                            endif()
                        endif()
                    endif()
                endif()
            endif()
        endif()
        
        set(filament_cxx_flags "${CMAKE_CXX_FLAGS} -Wno-deprecated" "-Wno-pass-failed=transform-warning" "-Wno-error=nonnull")
        if(NOT WIN32)
            set(filament_cxx_flags "${filament_cxx_flags} -fno-builtin")
        endif()
        
        # Build matc from source
        ExternalProject_Add(
            ext_filament_matc
            PREFIX filament-matc
            URL https://github.com/google/filament/archive/refs/tags/v1.58.2.tar.gz
            URL_HASH SHA256=8fbb35db77f34138e0c0536866d0de81e49b3ba2a3bd2d75e6de834779515cda
            DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/filament"
            UPDATE_COMMAND ""
            CMAKE_ARGS
                ${ExternalProject_CMAKE_ARGS}
                -DCMAKE_BUILD_TYPE=Release
                -DCCACHE_PROGRAM=OFF
                -DFILAMENT_ENABLE_JAVA=OFF
                -DCMAKE_C_COMPILER=${MATC_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${MATC_CXX_COMPILER}
                -DCMAKE_CXX_FLAGS:STRING=${filament_cxx_flags}
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DUSE_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}
                -DUSE_STATIC_LIBCXX=ON
                -DFILAMENT_SKIP_SDL2=ON
                -DFILAMENT_SKIP_SAMPLES=ON
                -DFILAMENT_BUILD_FILAMAT=ON
                -DFILAMENT_BUILD_FILABRIDGE=ON
                -DFILAMENT_BUILD_IMAGE=ON
                -DFILAMENT_BUILD_UTILS=ON
                -DFILAMENT_BUILD_TOOLS=ON
                -DFILAMENT_BUILD_FILAMENT=OFF
                -DFILAMENT_BUILD_GEOMETRY=OFF
                -DFILAMENT_BUILD_BACKEND=OFF
                -DFILAMENT_BUILD_GLSLANG=OFF
                -DFILAMENT_BUILD_SPIRV_TOOLS=OFF
                -DFILAMENT_BUILD_VALIDATOR=OFF
                -DFILAMENT_BUILD_SHADERS=OFF
                -DFILAMENT_BUILD_CMGEN=OFF
                -DFILAMENT_BUILD_FILAMESH=OFF
                -DFILAMENT_BUILD_FILAMESHIO=OFF
                -DFILAMENT_BUILD_KTXREADER=OFF
                -DFILAMENT_BUILD_MESHOPTIMIZER=OFF
                -DFILAMENT_BUILD_SMOLV=OFF
                -DFILAMENT_BUILD_VKSHADERS=OFF
                -DFILAMENT_BUILD_IBL=OFF
                -DFILAMENT_BUILD_BLUEGL=OFF
                -DFILAMENT_BUILD_BLUEVK=OFF
                -DSPIRV_WERROR=OFF
            BUILD_COMMAND ${CMAKE_COMMAND} --build . --target matc --config Release
            INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Skipping install"
            BUILD_BYPRODUCTS
                <BINARY_DIR>/tools/matc/matc${CMAKE_EXECUTABLE_SUFFIX}
            DEPENDS ext_filament
        )
        
        ExternalProject_Get_Property(ext_filament_matc BINARY_DIR)
        set(FILAMENT_MATC_BUILT "${BINARY_DIR}/tools/matc/matc${CMAKE_EXECUTABLE_SUFFIX}")
        message(STATUS "matc will be built at: ${FILAMENT_MATC_BUILT}")
    endif()
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")