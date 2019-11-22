set(CMAKE_C_FLAGS "-fPIC")
set(CMAKE_CXX_FLAGS "-std=c++14 -lc++ -lc++abi -fPIC")

file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")

include_directories(".")

add_executable(EncodeShader ${HEADER_FILES} ${SOURCE_FILES})
ShowAndAbortOnWarning(EncodeShader)

set_target_properties(EncodeShader PROPERTIES FOLDER "Tools")
