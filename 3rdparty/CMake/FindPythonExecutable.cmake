# Option 1: Do not define "PYTHON_EXECUTABLE", but run `cmake ..` within your
#           virtual environment. CMake will pick up the python executable in the
#           virtual environment.
# Option 2: You can also define `cmake -DPYTHON_EXECUTABLE` to specify a python
#           executable.
find_program(PYTHON_IN_PATH "python")
if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE ${PYTHON_IN_PATH} CACHE FILEPATH "Python exectuable to use")
    message(STATUS "Using python from PATH: ${PYTHON_EXECUTABLE}")
else()
    message(STATUS "Using python from PYTHON_EXECUTABLE variable: ${PYTHON_EXECUTABLE}")
    if ("${PYTHON_EXECUTABLE}" STREQUAL "${PYTHON_IN_PATH}")
        message(STATUS "(PYTHON_EXECUTABLE matches python from PATH)")
    else()
        message(STATUS "(PYTHON_EXECUTABLE does NOT match python from PATH: ${PYTHON_IN_PATH})")
    endif()
endif()
