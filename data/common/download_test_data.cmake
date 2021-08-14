# Download Open3D test data files. The default download path is
# Open3D/examples/test_data/open3d_downloads
#
# See https://github.com/isl-org/open3d_downloads for details on how to
# manage the test data files.


set(TEST_DATA_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(DOWNLOAD_DIR "${TEST_DATA_DIR}/open3d_downloads")


function(download_dataset_file dataset_url dataset_path dataset_sha256)
    set(DATASET_FULL_PATH "${DOWNLOAD_DIR}/${dataset_path}")
    get_filename_component(DATASET_FULL_PATH "${DATASET_FULL_PATH}" ABSOLUTE)

    # The saved file must be inside DOWNLOAD_DIR.
    if (NOT DATASET_FULL_PATH MATCHES "^${DOWNLOAD_DIR}")
        message(FATAL_ERROR "${DATASET_FULL_PATH} must be inside ${DOWNLOAD_DIR}")
    endif()

    # Support subdirectory inside TEST_DATA_DIR, e.g.
    # Open3D/examples/test_data/open3d_downloads/foo/bar/my_file.txt
    get_filename_component(DATASET_DIRECTORY "${DATASET_FULL_PATH}" DIRECTORY)
    file(MAKE_DIRECTORY "${DATASET_DIRECTORY}")

    if (NOT EXISTS "${DATASET_FULL_PATH}")
        message(STATUS "Downloading ${dataset_url} to ${DATASET_FULL_PATH}")
    endif()
    # Already downloaded files from previous are automatically ignored.
    file(DOWNLOAD "${dataset_url}" "${DATASET_FULL_PATH}"
        SHOW_PROGRESS
        EXPECTED_HASH SHA256=${dataset_sha256})
endfunction()


file(READ "${TEST_DATA_DIR}/download_file_list.json" DATASETS)

string(JSON DATASET_NUMBER LENGTH ${DATASETS})
if (NOT DATASET_NUMBER STREQUAL "0")
    # foreach includes the last element, so use N-1 instead of N
    math(EXPR DATASET_NUMBER_INCLUDED "${DATASET_NUMBER} - 1")
    foreach(index RANGE "${DATASET_NUMBER_INCLUDED}")
        string(JSON DATASET_NAME MEMBER ${DATASETS} ${index})
        string(JSON DATASET_URL GET ${DATASETS} ${DATASET_NAME} "url")
        string(JSON DATASET_PATH GET ${DATASETS} ${DATASET_NAME} "path")
        string(JSON DATASET_SHA256 GET ${DATASETS} ${DATASET_NAME} "sha256")

        download_dataset_file("${DATASET_URL}" "${DATASET_PATH}" "${DATASET_SHA256}")
    endforeach()
endif()
