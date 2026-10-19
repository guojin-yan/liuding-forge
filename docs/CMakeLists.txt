# Every installed public header participates in the strict documentation gate.
set(DOXYGEN_PROJECT_NAME "JYPPX Training Framework")
set(DOXYGEN_PROJECT_NUMBER "${PROJECT_VERSION}")
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_HIDE_UNDOC_MEMBERS NO)
set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
set(DOXYGEN_WARN_IF_DOC_ERROR YES)
set(DOXYGEN_WARN_IF_INCOMPLETE_DOC YES)
set(DOXYGEN_WARN_NO_PARAMDOC YES)
set(DOXYGEN_WARN_IF_UNDOC_ENUM_VAL YES)
set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}/include")

doxygen_add_docs(jyppx_api_docs
    "${PROJECT_SOURCE_DIR}/include/jyppx"
    COMMENT "Validating and generating complete public API documentation"
)

# Doxygen 1.18 on Windows can crash while replacing an existing HTML search
# index. Documentation is a reproducible build artifact, so always start this
# target from its own empty output directory on every generator and platform.
add_custom_target(jyppx_api_docs_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${DOXYGEN_OUTPUT_DIRECTORY}"
    COMMENT "Cleaning generated public API documentation"
    VERBATIM
)
add_dependencies(jyppx_api_docs jyppx_api_docs_clean)
