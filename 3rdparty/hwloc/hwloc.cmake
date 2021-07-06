include(ExternalProject)

include(ProcessorCount)
ProcessorCount(NPROC)

ExternalProject_Add(
    ext_hwloc
    PREFIX hwloc
    URL https://github.com/open-mpi/hwloc/archive/refs/tags/hwloc-2.5.0.tar.gz
    URL_HASH SHA256=67a2abfec135fca86e8aa3952bd0e77e3edb143ae88d67365b99b53e7e982eaa
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
    BUILD_COMMAND make -j${NPROC}
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}hwloc${CMAKE_STATIC_LIBRARY_SUFFIX}
    BUILD_IN_SOURCE ON
)

ExternalProject_Get_Property(ext_hwloc INSTALL_DIR)
set(HWLOC_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(HWLOC_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(HWLOC_LIBRARIES hwloc)
