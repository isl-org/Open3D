# MKL and TBB build scripts.
#
# This scripts exports: (both MKL and TBB)
# - STATIC_MKL_INCLUDE_DIR
# - STATIC_MKL_LIB_DIR
# - STATIC_MKL_LIBRARIES
#
# The name "STATIC" is used to avoid naming collisions for other 3rdparty CMake
# files (e.g. PyTorch) that also depends on MKL.

include(ExternalProject)

if(WIN32)
    set(MKL_INCLUDE_URL
        https://github.com/isl-org/Open3D/releases/download/v0.12.0/mkl-include-2020.1-intel_216-win-64.tar.bz2W
        https://anaconda.org/intel/mkl-include/2020.1/download/win-64/mkl-include-2020.1-intel_216.tar.bz2
    )
    set(MKL_INCLUDE_SHA256 65cedb770358721fd834224cd8be1fe1cc10b37ef2a1efcc899fc2fefbeb5b31)

    set(MKL_URL
        https://github.com/isl-org/Open3D/releases/download/v0.12.0/mkl-static-2020.1-intel_216-win-64.tar.bz2
        https://anaconda.org/intel/mkl-static/2020.1/download/win-64/mkl-static-2020.1-intel_216.tar.bz2
    )
    set(MKL_SHA256 c6f037aa9e53501d91d5245b6e65020399ebf34174cc4d03637818ebb6e6b6b9)
elseif(APPLE)
    set(MKL_INCLUDE_URL
        https://github.com/isl-org/Open3D/releases/download/v0.12.0/mkl-include-2020.1-intel_216-osx-64.tar.bz2
        https://anaconda.org/intel/mkl-include/2020.1/download/osx-64/mkl-include-2020.1-intel_216.tar.bz2
    )
    set(MKL_INCLUDE_SHA256 d4d025bd17ce75b92c134f70759b93ae1dee07801d33bcc59e40778003f05de5)

    set(MKL_URL
        https://github.com/isl-org/Open3D/releases/download/v0.12.0/mkl-static-2020.1-intel_216-osx-64.tar.bz2
        https://anaconda.org/intel/mkl-static/2020.1/download/osx-64/mkl-static-2020.1-intel_216.tar.bz2
    )
    set(MKL_SHA256 ca94ab8933cf58cbb7b42ac1bdc8671a948490fd1e0e9cea71a5b4d613b21be4)
else()
    set(MKL_INCLUDE_URL
        https://github.com/isl-org/Open3D/releases/download/v0.12.0/mkl-include-2020.1-intel_217-linux-64.tar.bz2
        https://anaconda.org/intel/mkl-include/2020.1/download/linux-64/mkl-include-2020.1-intel_217.tar.bz2
    )
    set(MKL_INCLUDE_SHA256 c0c4e7f261aa9182d811b91132c622211e55a5f3dfb8afb65a5377804f39eb61)

    set(MKL_URL
        https://github.com/isl-org/Open3D/releases/download/v0.12.0/mkl-static-2020.1-intel_217-linux-64.tar.bz2
        https://anaconda.org/intel/mkl-static/2020.1/download/linux-64/mkl-static-2020.1-intel_217.tar.bz2
    )
    set(MKL_SHA256 44fe60fa895c8967fe7c70fd1b680700f23ecac6ae038b267aa0a0c48dce3d59)

    # URL for merged libmkl_merged.a for Ubuntu.
    set(MKL_MERGED_URL
        https://github.com/isl-org/Open3D/releases/download/v0.10.0/linux-merged-mkl-static-2020.1-intel_217.zip
    )
    set(MKL_MERGED_SHA256 027c2b0d89c554479edbe5faecb93c26528877c1b682f939f8e1764d96860064)
endif()

# Where MKL and TBB headers and libs will be installed.
# This needs to be consistent with tbb.cmake.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)
set(STATIC_MKL_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/include/")
set(STATIC_MKL_LIB_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_LIB_DIR}")

if(WIN32)
    ExternalProject_Add(
        ext_mkl_include
        PREFIX mkl_include
        URL ${MKL_INCLUDE_URL}
        URL_HASH SHA256=${MKL_INCLUDE_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/Library/include ${MKL_INSTALL_PREFIX}/include
    )
    ExternalProject_Add(
        ext_mkl
        PREFIX mkl
        URL ${MKL_URL}
        URL_HASH SHA256=${MKL_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/Library/lib ${STATIC_MKL_LIB_DIR}
        BUILD_BYPRODUCTS
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_intel_ilp64${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_core${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_sequential${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_tbb_thread${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    # Generator expression can result in an empty string "", causing CMake to try to
    # locate ".lib". The workaround to first list all libs, and remove unneeded items
    # using generator expressions.
    set(STATIC_MKL_LIBRARIES
        mkl_intel_ilp64
        mkl_core
        mkl_sequential
        mkl_tbb_thread
        tbb_static
    )
    list(REMOVE_ITEM MKL_LIBRARIES "$<$<CONFIG:Debug>:mkl_tbb_thread>")
    list(REMOVE_ITEM MKL_LIBRARIES "$<$<CONFIG:Debug>:tbb_static>")
    list(REMOVE_ITEM MKL_LIBRARIES "$<$<CONFIG:Release>:mkl_sequential>")
elseif(APPLE)
    ExternalProject_Add(
        ext_mkl_include
        PREFIX mkl_include
        URL ${MKL_INCLUDE_URL}
        URL_HASH SHA256=${MKL_INCLUDE_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${MKL_INSTALL_PREFIX}/include
    )
    ExternalProject_Add(
        ext_mkl
        PREFIX mkl
        URL ${MKL_URL}
        URL_HASH SHA256=${MKL_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/lib ${STATIC_MKL_LIB_DIR}
        BUILD_BYPRODUCTS
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_intel_ilp64${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_tbb_thread${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_core${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    set(STATIC_MKL_LIBRARIES mkl_intel_ilp64 mkl_tbb_thread mkl_core tbb_static)
else()
    ExternalProject_Add(
        ext_mkl_include
        PREFIX mkl_include
        URL ${MKL_INCLUDE_URL}
        URL_HASH SHA256=${MKL_INCLUDE_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${MKL_INSTALL_PREFIX}/include
    )
    option(USE_LINUX_MKL_FROM_CONDA_REPO "On linux, use MKL from official conda repo" OFF)
    if(USE_LINUX_MKL_FROM_CONDA_REPO)
        # Resolving static library circular dependencies.
        # - Approach 1: Add `-Wl,--start-group` `-Wl,--end-group` around, but this
        #               is not friendly with CMake.
        # - Approach 2: Set LINK_INTERFACE_MULTIPLICITY to 3. However this does not
        #               work directly with interface library, and requires big
        #               changes to the build system. See discussions in:
        #               - https://gitlab.kitware.com/cmake/cmake/-/issues/17964
        #               - https://gitlab.kitware.com/cmake/cmake/-/issues/18415
        #               - https://stackoverflow.com/q/50166553/1255535
        # - Approach 3: Merge libmkl_intel_ilp64.a, libmkl_tbb_thread.a and
        #               libmkl_core.a into libmkl_merged.a. This is the most simple
        #               approach to integrate with the build system. However, extra
        #               time is required to merge the libraries and the merged
        #               library size can be large. We choose to use approach 3.
        ExternalProject_Add(
            ext_mkl
            PREFIX mkl
            URL ${MKL_URL}
            URL_HASH SHA256=${MKL_SHA256}
            DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_IN_SOURCE ON
            BUILD_COMMAND echo "Extracting static libs..."
            COMMAND ar x lib/libmkl_intel_ilp64.a
            COMMAND ar x lib/libmkl_tbb_thread.a
            COMMAND ar x lib/libmkl_core.a
            COMMAND echo "Merging static libs..."
            COMMAND bash -c "ar -qc lib/libmkl_merged.a *.o"
            COMMAND echo "Cleaning up *.o files..."
            COMMAND bash -c "rm *.o"
            INSTALL_COMMAND ${CMAKE_COMMAND} -E copy lib/libmkl_merged.a ${STATIC_MKL_LIB_DIR}/libmkl_merged.a
            BUILD_BYPRODUCTS ${STATIC_MKL_LIB_DIR}/libmkl_merged.a
        )
    else()
        # We also provide a direct download for libmkl_merged.a.
        ExternalProject_Add(
            ext_mkl
            PREFIX mkl
            URL ${MKL_MERGED_URL}
            URL_HASH SHA256=${MKL_MERGED_SHA256}
            DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_IN_SOURCE ON
            BUILD_COMMAND ""
            INSTALL_COMMAND ${CMAKE_COMMAND} -E copy lib/libmkl_merged.a ${STATIC_MKL_LIB_DIR}/libmkl_merged.a
            BUILD_BYPRODUCTS ${STATIC_MKL_LIB_DIR}/libmkl_merged.a
        )
    endif()
    set(STATIC_MKL_LIBRARIES mkl_merged tbb_static)
endif()
