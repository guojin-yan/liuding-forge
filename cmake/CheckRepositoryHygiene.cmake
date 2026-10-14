if(NOT DEFINED JYPPX_SOURCE_DIR)
    message(FATAL_ERROR "JYPPX_SOURCE_DIR is required")
endif()

file(REAL_PATH "${JYPPX_SOURCE_DIR}" source_root)
set(violations)

file(GLOB root_entries RELATIVE "${source_root}" "${source_root}/*")
foreach(entry IN LISTS root_entries)
    set(path "${source_root}/${entry}")
    if(IS_DIRECTORY "${path}" AND entry MATCHES "^[0-9]+$")
        list(APPEND violations "top-level numeric directory: ${entry}/")
    endif()
    if(IS_DIRECTORY "${path}" AND
       (entry MATCHES "^build($|[-_].*)$" OR entry MATCHES "^cmake-build-"))
        list(APPEND violations
             "legacy top-level build directory (use out/build/<preset>): ${entry}/")
    endif()
    if(NOT IS_DIRECTORY "${path}" AND
       entry MATCHES "\.(log|out|err|tmp|bak)$")
        list(APPEND violations "root runtime artifact: ${entry}")
    endif()
endforeach()

execute_process(
    COMMAND git ls-files
    WORKING_DIRECTORY "${source_root}"
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE tracked_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE git_error
)
if(NOT git_result EQUAL 0)
    message(FATAL_ERROR "git ls-files failed: ${git_error}")
endif()

string(REPLACE "\r\n" "\n" tracked_output "${tracked_output}")
string(REPLACE "\n" ";" tracked_files "${tracked_output}")
foreach(path IN LISTS tracked_files)
    if(path MATCHES "^(build[^/]*/|out/|Testing/|data/|runs/|artifacts/|test-output/)")
        list(APPEND violations "tracked generated/runtime path: ${path}")
    endif()
    if(path MATCHES "^[^/]+\.(log|out|err|tmp|bak)$")
        list(APPEND violations "tracked root runtime artifact: ${path}")
    endif()
endforeach()

set(max_fixture_bytes 5242880)
foreach(test_data_root IN ITEMS tests/fixtures tests/reference_data)
    if(EXISTS "${source_root}/${test_data_root}")
        file(GLOB_RECURSE fixture_files LIST_DIRECTORIES false
             "${source_root}/${test_data_root}/*")
        foreach(fixture IN LISTS fixture_files)
            file(SIZE "${fixture}" fixture_size)
            if(fixture_size GREATER max_fixture_bytes)
                file(RELATIVE_PATH relative_fixture "${source_root}" "${fixture}")
                list(APPEND violations
                     "test data exceeds 5 MiB without external manifest: ${relative_fixture}")
            endif()
        endforeach()
    endif()
endforeach()

if(violations)
    list(JOIN violations "\n  - " violation_text)
    message(FATAL_ERROR
        "Repository hygiene violations:\n  - ${violation_text}\n"
        "See docs/development/repository_hygiene.md")
endif()

message(STATUS "Repository hygiene check passed")
