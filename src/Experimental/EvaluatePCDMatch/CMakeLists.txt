get_filename_component(EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

file(GLOB SOURCE_FILES "*.cpp")

add_executable(${EXE_NAME} ${SOURCE_FILES})

target_link_libraries(${EXE_NAME}
                      Visualization
                      IO
                      Core)

set_target_properties(${EXE_NAME} PROPERTIES
                      FOLDER "Experimental"
                      RUNTIME_OUTPUT_DIRECTORY
                      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
