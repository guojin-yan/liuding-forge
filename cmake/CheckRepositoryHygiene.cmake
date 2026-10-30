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
    COMMAND git -c core.quotepath=false ls-files --cached --others --exclude-standard
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
list(REMOVE_DUPLICATES tracked_files)

# Only byte-exact, explicitly listed official generator output may omit the
# product header. The complete directory must match; it is never skipped.
set(generated_root "src/jyppx/liuding/telemetry_native/detail/telemetry_proto")
set(generated_manifest "${generated_root}/generated.sha256")
set(generated_files)
set(verified_generated_files)
# Derive output names from the existing locked family inputs, without requiring
# string(JSON), which is newer than the supported CMake baseline.
set(locked_generated_files)
file(READ "${source_root}/scripts/telemetry/native_schema.lock.json" schema_lock)
string(FIND "${schema_lock}" "\"families\"" schema_families_offset)
if(schema_families_offset EQUAL -1)
    list(APPEND violations "official schema family lock is missing")
else()
    string(SUBSTRING "${schema_lock}" ${schema_families_offset} -1 schema_families)
    string(REGEX MATCHALL "\"path\"[ \t\r\n]*:[ \t\r\n]*\"[A-Za-z0-9_./-]+[.]proto\""
        schema_paths "${schema_families}")
    foreach(schema_path IN LISTS schema_paths)
        string(REGEX REPLACE "^\"path\"[ \t\r\n]*:[ \t\r\n]*\"([^\"]+)\"$" "\\1"
            schema_path "${schema_path}")
        foreach(suffix IN ITEMS c h)
            string(REGEX REPLACE "[.]proto$" ".pb.${suffix}" generated_path "${schema_path}")
            list(APPEND locked_generated_files "${generated_root}/${generated_path}")
        endforeach()
    endforeach()
    list(SORT locked_generated_files)
endif()
if(NOT EXISTS "${source_root}/${generated_manifest}")
    list(APPEND violations "missing official generated manifest: ${generated_manifest}")
else()
    file(STRINGS "${source_root}/${generated_manifest}" generated_rows ENCODING UTF-8)
    foreach(row IN LISTS generated_rows)
        if(row MATCHES "^#" OR row STREQUAL "")
            continue()
        endif()
        if(NOT row MATCHES "^([0-9a-f]+)  ((tensorboard|opentelemetry)/[A-Za-z0-9_./-]+[.]pb[.][ch])$")
            list(APPEND violations "malformed official generated manifest row: ${row}")
            continue()
        endif()
        set(expected_digest "${CMAKE_MATCH_1}")
        set(relative_generated "${CMAKE_MATCH_2}")
        string(LENGTH "${expected_digest}" digest_length)
        if(NOT digest_length EQUAL 64 OR relative_generated MATCHES "(^|/)[.][.]?(/|$)|//")
            list(APPEND violations "unsafe official generated manifest row: ${row}")
            continue()
        endif()
        set(generated_path "${generated_root}/${relative_generated}")
        list(FIND generated_files "${generated_path}" duplicate_generated)
        if(NOT duplicate_generated EQUAL -1)
            list(APPEND violations "duplicate official generated manifest path: ${generated_path}")
            continue()
        endif()
        list(APPEND generated_files "${generated_path}")
        if(NOT EXISTS "${source_root}/${generated_path}" OR
           IS_DIRECTORY "${source_root}/${generated_path}" OR
           IS_SYMLINK "${source_root}/${generated_path}")
            list(APPEND violations "missing or nonregular official generated source: ${generated_path}")
            continue()
        endif()
        file(SHA256 "${source_root}/${generated_path}" actual_digest)
        file(READ "${source_root}/${generated_path}" generated_header LIMIT 256)
        if(NOT actual_digest STREQUAL expected_digest OR
           NOT generated_header MATCHES "^/[*] Automatically generated nanopb (constant definitions|header) [*]/[\r\n]+/[*] Generated by nanopb-0[.]4[.]9[.]2 [*]/")
            list(APPEND violations "official generated identity or marker mismatch: ${generated_path}")
            continue()
        endif()
        list(APPEND verified_generated_files "${generated_path}")
    endforeach()
    file(GLOB_RECURSE actual_generated_files LIST_DIRECTORIES false
        RELATIVE "${source_root}" "${source_root}/${generated_root}/*")
    list(REMOVE_ITEM actual_generated_files "${generated_manifest}")
    list(SORT actual_generated_files)
    list(SORT generated_files)
    if(NOT generated_files OR NOT "${generated_files}" STREQUAL "${locked_generated_files}")
        list(APPEND violations "official generated manifest differs from locked schema paths")
    endif()
    if(NOT generated_files OR NOT "${actual_generated_files}" STREQUAL "${generated_files}")
        list(APPEND violations "official generated file set differs from manifest: ${generated_root}")
    endif()
endif()
set(retired_placeholder_paths
    docs/api/distributed.md
    include/jyppx/liuding/distributed/ddp.hpp
    include/jyppx/liuding/distributed/distributed.hpp
    src/jyppx/liuding/distributed/distributed.cpp
)
foreach(path IN LISTS tracked_files)
    if(path MATCHES "^(include|src|examples|tests|tools)/.*[.](c|cc|cpp|cxx|cu|cuh|h|hh|hpp|hxx|inc|ipp|tpp)$")
        list(FIND verified_generated_files "${path}" official_generated_index)
        if(official_generated_index EQUAL -1)
            file(READ "${source_root}/${path}" source_header LIMIT 1024)
            if(NOT source_header MATCHES "Product: 六丁炼算 [(]Liuding Forge[)]" OR
               NOT source_header MATCHES "Project alias: JYPPX Training Framework")
                list(APPEND violations "missing product/project alias in source header: ${path}")
            endif()
        endif()
    endif()
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
