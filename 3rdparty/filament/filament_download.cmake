include(FetchContent)

set(filament_LIBRARIES filameshio filament filamat_lite filaflat filabridge geometry backend bluegl bluevk ibl image meshoptimizer smol-v utils vkshaders)

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
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.15.1/filament-v1.15.1-windows.tgz)
        set(FILAMENT_SHA256 3fdfde669ae4e1df8dccbcb149ec3c4248e88ea1758345d1e2e288a5278522ba)
        if (STATIC_WINDOWS_RUNTIME)
            string(APPEND lib_dir /x86_64/mt)
        else()
            string(APPEND lib_dir /x86_64/md)
        endif()
    elseif(APPLE)
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.15.1/filament-v1.15.1-mac.tgz)
        set(FILAMENT_SHA256 2c48431c37d1f9c67c81949441f3d6f593c495c235525b32e4e67f826c268ca9)
        string(APPEND lib_dir /x86_64)
    else()      # Linux: Check glibc version and use open3d filament binary if new (Ubuntu 20.04 and similar)
        set(FILAMENT_URL https://github.com/isl-org/open3d_downloads/releases/download/filament/filament-v1.15.1-linux.tgz)
        set(FILAMENT_SHA256 005d6ea95b2f84cb5bbdb1f3882ea7be775bbac95c7656360489aa20ada6ec79)
        string(APPEND lib_dir /x86_64)
        # NOTE: Uncertain if 20.04 will need its own version so leaving below in
        # for now
#        execute_process(COMMAND ldd --version OUTPUT_VARIABLE ldd_version)
        #string(REGEX MATCH "([0-9]+\.)+[0-9]+" glibc_version ${ldd_version})
        #if(${glibc_version} VERSION_LESS "2.31")
            #set(FILAMENT_URL
                #https://github.com/isl-org/open3d_downloads/releases/download/filament/filament-v1.9.19-linux.tgz)
            #set(FILAMENT_SHA256 f0c0b05a543dd0c82b1cd571957a90f28e72cfeee36d19a527c17ac9de4733d5)
            #message(STATUS "GLIBC version ${glibc_version} found: Using "
                #"Google Filament binary.")
        #else()
            #set(FILAMENT_URL
                #https://github.com/isl-org/open3d_downloads/releases/download/filament/filament-v1.9.19-linux-20.04.tgz)
            #set(FILAMENT_SHA256 c756fd76f5c6a40ca554f8c3cca424354a2a22ea6fce3c8ea893d4c4aa39514c)
            #message(STATUS "GLIBC version ${glibc_version} found: Using "
                #"Open3D Filament binary.")
#        endif()
    endif()

    set(lib_byproducts ${filament_LIBRARIES})
    list(TRANSFORM lib_byproducts PREPEND <SOURCE_DIR>/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
    list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})

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
    set(FILAMENT_ROOT ${SOURCE_DIR})
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")
