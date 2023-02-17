# open3d_make_hardening_flags(hardening_cflags hardening_ldflags)
#
# Sets up hardening compiler options and linker options and stores them
# into the <hardening_cflags> and <hardening_ldflags> variables.
function(open3d_make_hardening_flags hardening_cflags hardening_ldflags)
    unset(${hardening_cflags})
    unset(${hardening_ldflags})

    # -Wall -Wextra -Werror or /W4 /WX are enabled for Open3D code (not 3rd party)
    if (MSVC)
        set(${hardening_cflags}
            /sdl            # SDL Checks
            /GS             # Code Generation: Security Check
            /guard:cf       # Code Generation: Control Flow Guard
        )
        set(${hardening_ldflags}
            /INCREMENTAL:NO  # Disable incremental Linking
            /NXCOMPAT        # Data Execution Prevention: On by default in VS2019
            /DYNAMICBASE     # Randomized Base Address
            /HIGHENTROPYVA   #
            #/INTEGRITYCHECK # Signed binary: Disabled
        )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(${hardening_cflags}
            -fstack-protector               # Stack-based buffer overrun detection
            -Wformat -Wformat-security      # Format string vulnerability
        )
        set(${hardening_ldflags}
            -fsanitize=safe-stack       # Stack execution protection
            -Wl,-z,relro,-z,now         # Data relocation protection
            -pie                        # Position independent executable
            $<$<CONFIG:Release>:LINKER:-S>  # Exclude debug info
        )
        if(NOT BUILD_SHARED_LIBS AND NOT BUILD_PYTHON_MODULE)
            list(APPEND ${hardening_cflags} -fsanitize=safe-stack)   # Stack execution protection
            list(APPEND ${hardening_ldflags} -fsanitize=safe-stack)  # only static libraries supported
        endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(${hardening_cflags}
            -fstack-protector               # Stack-based buffer overrun detection
            -Wformat -Wformat-security      # Format string vulnerability
        )
        set(${hardening_ldflags}
            -Wl,-z,relro,-z,now             # Data relocation protection
            $<$<CONFIG:Release>:LINKER:-S>  # Exclude debug info
        )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(${hardening_cflags}
            -fstack-protector               # Stack-based buffer overrun detection
            -Wformat -Wformat-security      # Format string vulnerability
        )
        set(${hardening_ldflags}
            # -pie Position independent executable is default on macOSX 10.6+
            LINKER:-dead_strip    # Remove unreachable code
            $<$<CONFIG:Release>:LINKER:-S,-x>  # Exclude debug info, non-global symbols
        )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(${hardening_cflags}
            -fstack-protector-strong    # Stack-based buffer overrun detection
            -Wformat -Wformat-security  # Format string vulnerability
        )
        set(${hardening_ldflags}
            -Wl,-z,noexecstack   # Stack execution protection
            -Wl,-z,relro,-z,now  # Data relocation protection
            -pie                 # Position independent executable
            $<$<CONFIG:Release>:LINKER:--strip-debug>  # Exclude debug info
        )
    else()
        message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. No security "
        "flags set")
    endif()

    # Remove unsupported compiler flags
    include(CheckCXXCompilerFlag)
    foreach(flag IN LISTS ${hardening_cflags})
        string(MAKE_C_IDENTIFIER "${flag}" FLAGRESULT)
        check_cxx_compiler_flag("${flag}" FLAG${FLAGRESULT})
        if (NOT FLAG${FLAGRESULT})
            list(REMOVE_ITEM ${hardening_cflags} ${flag})
            message(WARNING "Compiler does not support security option ${flag}")
        endif()
    endforeach()

    # Remove unsupported linker flags
    include(CheckLinkerFlag)
    foreach(flag IN LISTS ${hardening_ldflags})
        string(MAKE_C_IDENTIFIER "${flag}" FLAGRESULT)
        check_linker_flag(CXX "${flag}" FLAG${FLAGRESULT})
        if (NOT FLAG${FLAGRESULT})
            list(REMOVE_ITEM ${hardening_ldflags} ${flag})
            message(WARNING "Linker does not support security option ${flag}")
        endif()
    endforeach()

    list(TRANSFORM ${hardening_ldflags} REPLACE "-pie"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>")
    string(REPLACE ";" "," cuda_hardening_cflags "${${hardening_cflags}}")
    string(REPLACE ";" "," cuda_hardening_ldflags "${${hardening_ldflags}}")

    set(${hardening_cflags}
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${cuda_hardening_cflags}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${${hardening_cflags}}>"
    )
    set(${hardening_ldflags}
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${cuda_hardening_ldflags}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${${hardening_ldflags}}>"
    )

    set(${hardening_cflags} ${${hardening_cflags}} PARENT_SCOPE)
    set(${hardening_ldflags} ${${hardening_ldflags}} PARENT_SCOPE)
endfunction()


# open3d_make_hardening_definitions(hardening_definitions)
#
# Makes hardening compiler definitions and stores them into the <hardening_definitions> variable.
function(open3d_make_hardening_definitions hardening_definitions)
    unset(${hardening_definitions})

    if (MSVC)
        # No flags added for MSVC
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
            CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
            CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
            CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(${hardening_definitions} _FORTIFY_SOURCE=2)     # Buffer overflow detection
    else()
        message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. No security "
        "flags set")
    endif()

    set(${hardening_definitions} ${${hardening_definitions}} PARENT_SCOPE)
endfunction()
