if(NOT DEFINED JYPPX_SOURCE_DIR)
    message(FATAL_ERROR "JYPPX_SOURCE_DIR is required")
endif()

file(REAL_PATH "${JYPPX_SOURCE_DIR}" source_root)
set(violations)

set(required_legal_files LICENSE NOTICE THIRD_PARTY_NOTICES.md)
foreach(legal_file IN LISTS required_legal_files)
    if(NOT EXISTS "${source_root}/${legal_file}")
        list(APPEND violations "missing required legal file: ${legal_file}")
    endif()
endforeach()

if(EXISTS "${source_root}/LICENSE")
    file(READ "${source_root}/LICENSE" project_license)
    if(NOT project_license MATCHES "Apache License[\r\n ]+Version 2\\.0")
        list(APPEND violations "LICENSE is not the Apache License, Version 2.0")
    endif()
    if(project_license MATCHES "MIT License")
        list(APPEND violations "LICENSE contains a second project license")
    endif()
endif()

foreach(metadata_file IN ITEMS README.md CONTRIBUTING.md docs/index.md)
    if(EXISTS "${source_root}/${metadata_file}")
        file(READ "${source_root}/${metadata_file}" metadata_text)
        if(NOT metadata_text MATCHES "Apache(-| )License(,)?( Version)? 2\\.0|Apache-2\\.0")
            list(APPEND violations
                 "current project license is not declared as Apache-2.0: ${metadata_file}")
        endif()
        if(metadata_text MATCHES "licensed under the MIT License|License: MIT|MIT License -")
            list(APPEND violations
                 "stale MIT project-license declaration: ${metadata_file}")
        endif()
    endif()
endforeach()

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
       entry MATCHES "[.](log|out|err|tmp|bak)$")
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
set(retired_placeholder_paths
    docs/api/distributed.md
    include/jyppx/distributed/ddp.hpp
    include/jyppx/distributed/distributed.hpp
    src/jyppx/distributed/distributed.cpp
)
foreach(path IN LISTS tracked_files)
    list(FIND retired_placeholder_paths "${path}" retired_placeholder_index)
    if(NOT retired_placeholder_index EQUAL -1)
        list(APPEND violations
             "retired v1.0 distributed placeholder: ${path}")
    endif()
    if(path MATCHES "^(build[^/]*/|out/|Testing/|data/|runs/|artifacts/|test-output/)")
        list(APPEND violations "tracked generated/runtime path: ${path}")
    endif()
    if(path MATCHES "^[^/]+[.](log|out|err|tmp|bak)$")
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
