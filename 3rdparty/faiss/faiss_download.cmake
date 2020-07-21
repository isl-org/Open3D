include(FetchContent)

if (FAISS_PRECOMPILED_ROOT)
    if (EXISTS "${FAISS_PRECOMPILED_ROOT}")
        set(FAISS_ROOT "${FAISS_PRECOMPILED_ROOT}")
    else()
        message(FATAL_ERROR "Faiss binaries not found in ${FAISS_PRECOMPILED_ROOT}")
    endif()
else()
    # Setup download links
    if(WIN32)
        set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/faiss-20200127-windows.tgz")
        set(DOWNLOAD_URL_FALLBACK "https://github.com/google/faiss/releases/download/v1.4.5/faiss-20200127-windows.tgz")
    elseif(APPLE)
        set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/faiss-20200127-mac-10.14-resizefix2.tgz")
        set(DOWNLOAD_URL_FALLBACK "https://github.com/google/faiss/releases/download/v1.4.5/faiss-20200127-mac.tgz")
    else()
        set(DOWNLOAD_URL_PRIMARY "https://github.com/junha-l/faiss_tmp/releases/download/0.1/faiss-cuda10.0.130-20200721-linux.tgz")
        set(DOWNLOAD_URL_FALLBACK "https://github.com/junha-l/faiss_tmp/releases/download/0.1/faiss-20200720-linux.tgz")
    endif()

    #if (USE_VULKAN AND (ANDROID OR WIN32 OR WEBGL OR IOS))
    #    MESSAGE(FATAL_ERROR "Downloadable version of Faiss supports vulkan only on Linux and Apple")
    #endif()

    FetchContent_Declare(
        fetch_faiss
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
    )

    message(STATUS "${DOWNLOAD_URL_FALLBACK} ${DOWNLOAD_URL_PRIMARY}")
    # FetchContent happends at config time.
    FetchContent_GetProperties(fetch_faiss)
    if(NOT fetch_faiss_POPULATED)
        message(STATUS "Downloading Faiss...")
        FetchContent_Populate(fetch_faiss)
        # We use the default download and unpack directories for FetchContent.
        message(STATUS "Faiss has been downloaded to ${fetch_faiss_DOWNLOADED_FILE}.")
        message(STATUS "Faiss has been extracted to ${fetch_faiss_SOURCE_DIR}.")
        set(FAISS_ROOT "${fetch_faiss_SOURCE_DIR}")
    endif()
endif()

message(STATUS "Faiss is located at ${FAISS_ROOT}")

set(FAISS_LIBRARIES "faiss")
#if (UNIX)
#    set(faiss_LIBRARIES ${faiss_LIBRARIES} bluevk)
#endif()
