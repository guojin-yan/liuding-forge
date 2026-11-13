# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# Install only the provenance of the foundation archive that was actually built.
# Configure-time candidates and successful-build stamps must stay separate.
function(jyppx_install_build_provenance target)
    if(NOT DEFINED JYPPX_LIUDING_VERSION)
        set(JYPPX_LIUDING_VERSION "${PROJECT_VERSION}")
    endif()
    set(provenance_root "${CMAKE_CURRENT_BINARY_DIR}/build-provenance")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/jyppx_build_provenance.json.in"
        "${provenance_root}/configured.json.in"
        @ONLY
    )
    file(GENERATE
        OUTPUT "${provenance_root}/configured/$<CONFIG>/build-provenance.json"
        INPUT "${provenance_root}/configured.json.in"
    )
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${provenance_root}/built/$<CONFIG>"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${provenance_root}/configured/$<CONFIG>/build-provenance.json"
            "${provenance_root}/built/$<CONFIG>/build-provenance.json"
        COMMENT "Recording successful foundation build provenance"
        VERBATIM
    )
    # No OPTIONAL: an unbuilt configuration must fail rather than acquire a
    # configure-time identity. Reconfiguring also leaves prior stamps intact.
    install(FILES "${provenance_root}/built/$<CONFIG>/build-provenance.json"
        DESTINATION share/jyppx
    )
endfunction()
