# Assert that we're inside a conda environemnt
message(STATUS "Checking Conda environment...")
execute_process(
    COMMAND bash "-c" "conda info | grep -q \"active environment : None\""
    RESULT_VARIABLE return_code
)

if (${return_code})
    message("Conda environment is active.")
else()
    message(FATAL_ERROR "Not in a conda environment: 1) activate conda, 2) rerun cmake and make")
endif()

# Check if conda-build is installed, if not, install it
find_program(CONDA_BUILD "conda-build")
if (NOT CONDA_BUILD)
    message(STATUS "conda-build not found")
    message(STATUS "Trying to install conda-build ...")
    execute_process(
        COMMAND conda install -y -q conda-build
        RESULT_VARIABLE return_code
    )
    find_program(CONDA_BUILD "conda-build")
    if (NOT CONDA_BUILD)
        message(FATAL_ERROR "conda-build installation failed, please install manualy")
    else()
        message(STATUS "conda-build installed at ${CONDA_BUILD}")
    endif()
endif()
