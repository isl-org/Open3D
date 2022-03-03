# open3d_add_resources_to_binary(<target>
#    RESOURCES_LIST <resources>
# )
#
# Converts the files present in the resources folder <dir> into binaries that can be embedded into source files.
function(open3d_add_resources_to_binary out_var target_var)
  set(result)
  set(curdir ${PROJECT_BINARY_DIR}/bin/resources)
  # file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  # set(filelist "")
  # foreach(child ${children})
  #   if(NOT IS_DIRECTORY ${curdir}/${child})
  #     list(APPEND filelist ${child})
  #   endif()
  # endforeach()
  file(MAKE_DIRECTORY ${curdir}/bin)
  foreach(in_f ${ARGN})
    set(out_f "${curdir}/bin/${in_f}.o")
    # file(TOUCH ${out_f})
    message("ld -r -b binary -o ${out_f} ${in_f}")
    add_custom_command(TARGET ${target_var}
    PRE_LINK
      # OUTPUT ${out_f}
    COMMAND ld -r -b binary -o ${out_f} ${in_f} && echo ${out_f} ${in_f}
    DEPENDS ${curdir}/${in_f}
    WORKING_DIRECTORY ${curdir}
    COMMENT "Building ${in_f} to ${out_f}"
    VERBATIM
    )
    list(APPEND result ${out_f})
    message("${in_f}\n${out_f}")
    endforeach()
  list(GET result 1 first)
  # add_custom_target(resources_target ALL
  # COMMAND echo "This is ALL target 'zoo', and it depends on ${first}"
  # DEPENDS ${first}
  # VERBATIM
  # )
  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()
