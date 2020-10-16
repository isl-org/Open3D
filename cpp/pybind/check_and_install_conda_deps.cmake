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
