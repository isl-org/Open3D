include(ExternalProject)

include(ProcessorCount)
ProcessorCount(NPROC)

if(WIN32)
    set(HWLOC_BUILD_FROM_SOURCE OFF)
elseif(APPLE)
    # `brew install automake` to install these dependencies
    find_program(AUTOCONF_BIN NAMES autoconf)
    find_program(AUTOMAKE_BIN NAMES automake)
    find_program(LIBTOOL_BIN NAMES libtool)
    if(AUTOCONF_BIN AND AUTOMAKE_BIN AND LIBTOOL_BIN)
        set(HWLOC_BUILD_FROM_SOURCE ON)
        set(HWLOC_URL https://github.com/open-mpi/hwloc/archive/refs/tags/hwloc-2.5.0.tar.gz)
        set(HWLOC_SHA256 67a2abfec135fca86e8aa3952bd0e77e3edb143ae88d67365b99b53e7e982eaa)
    else()
        set(HWLOC_BUILD_FROM_SOURCE OFF)
        set(HWLOC_URL https://github.com/intel-isl/open3d_downloads/releases/download/hwloc-macos/hwloc-macos-10.14-build-2.5.0.zip)
        set(HWLOC_SHA256 d6c48ab1bf515cf631d98a6800ce4e86c3d1281f215db54ce4304b14103d61ee)
    endif()
else()
    set(HWLOC_BUILD_FROM_SOURCE ON)
    set(HWLOC_URL https://github.com/open-mpi/hwloc/archive/refs/tags/hwloc-2.5.0.tar.gz)
    set(HWLOC_SHA256 67a2abfec135fca86e8aa3952bd0e77e3edb143ae88d67365b99b53e7e982eaa)
endif()

if(HWLOC_BUILD_FROM_SOURCE)
    ExternalProject_Add(
        ext_hwloc
        PREFIX hwloc
        URL ${HWLOC_URL}
        URL_HASH SHA256=${HWLOC_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/hwloc"
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ./autogen.sh
        COMMAND ./configure --prefix=<INSTALL_DIR>
                            --enable-silent-rules
                            --enable-static
                            --silent
                            --with-pic
                            --disable-dependency-tracking
                            --disable-picky
                            --disable-cairo
                            --disable-libxml2
                            --disable-io
                            --disable-pci
                            --disable-opencl
                            --disable-cuda
                            --disable-nvml
                            --disable-rsmi
                            --disable-levelzero
                            --disable-gl
                            --disable-libudev
                            --disable-plugin-dlopen
                            --disable-plugin-ltdl
        BUILD_COMMAND make CFLAGS=$<$<PLATFORM_ID:Darwin>:-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}> -j${NPROC}
        INSTALL_COMMAND make install
        BUILD_BYPRODUCTS
            <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}hwloc${CMAKE_STATIC_LIBRARY_SUFFIX}
        BUILD_IN_SOURCE ON
    )

    ExternalProject_Get_Property(ext_hwloc INSTALL_DIR)
    set(HWLOC_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    set(HWLOC_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
    set(HWLOC_LIBRARIES hwloc)
else()
    ExternalProject_Add(
        ext_hwloc
        PREFIX hwloc
        URL ${HWLOC_URL}
        URL_HASH SHA256=${HWLOC_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/hwloc"
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )

    ExternalProject_Get_Property(ext_hwloc SOURCE_DIR)
    set(HWLOC_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.
    set(HWLOC_LIB_DIR ${SOURCE_DIR}/lib)
    set(HWLOC_LIBRARIES hwloc)
endif()
