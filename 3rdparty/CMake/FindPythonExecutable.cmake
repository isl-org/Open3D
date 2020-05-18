# Option 1: Do not define "PYTHON_EXECUTABLE", but run `cmake ..` within your
#           virtual environment. CMake will pick up the python executable in the
#           virtual environment.
# Option 2: You can also define `cmake -DPYTHON_EXECUTABLE` to specify a python
#           executable.
if (NOT PYTHON_EXECUTABLE)
    # find_program will returns the python executable in current PATH, which
    # works with virtualenv
    find_program(PYTHON_IN_PATH "python")
    set(PYTHON_EXECUTABLE ${PYTHON_IN_PATH})
endif()
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")
