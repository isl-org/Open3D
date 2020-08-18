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
        message(FATAL_ERROR "Faiss is not supported on Windonws")
    elseif(APPLE)
        set(DOWNLOAD_URL_PRIMARY "https://github.com/junha-l/faiss_tmp/releases/download/0.1/faiss-cpu-20200805-osx.tgz")
    else()
        if (BUILD_CUDA_MODULE)
            set(DOWNLOAD_URL_PRIMARY "https://github.com/junha-l/faiss_tmp/releases/download/0.1/faiss-cuda10.1.243-20200818-linux.tgz")
        else()
            set(DOWNLOAD_URL_PRIMARY "https://github.com/junha-l/faiss_tmp/releases/download/0.1/faiss-cpu-20200818-linux.tgz")
        endif()
    endif()

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
