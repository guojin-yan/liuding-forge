# Compiler Options for JYPPX Training Framework
#
# Per-configuration flags use generator expressions so they work with both
# single-config (Ninja/Make) and multi-config (Visual Studio) generators.

# ============================================
# Warning Flags
# ============================================
set(JYPPX_COMPILED_TARGETS jyppx_foundation jyppx_training_framework)
foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${JYPPX_TARGET} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wno-unused-parameter
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2>
            $<$<CONFIG:Debug>:-O0;-g>
        )
    elseif(MSVC)
        target_compile_options(${JYPPX_TARGET} PRIVATE
            /W4
            /wd4100  # Unused parameter
        )
    endif()
endforeach()

# ============================================
# Debug Definitions
# ============================================
foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
    target_compile_definitions(${JYPPX_TARGET} PRIVATE
        $<$<CONFIG:Debug>:JYPPX_DEBUG>
    )
endforeach()

# ============================================
# Sanitizers (Optional)
# ============================================
option(JYPPX_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(JYPPX_ENABLE_TSAN "Enable Thread Sanitizer" OFF)

if(JYPPX_ENABLE_ASAN AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
        target_compile_options(${JYPPX_TARGET} PRIVATE -fsanitize=address)
        target_link_options(${JYPPX_TARGET} PRIVATE -fsanitize=address)
    endforeach()
endif()

if(JYPPX_ENABLE_TSAN AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
        target_compile_options(${JYPPX_TARGET} PRIVATE -fsanitize=thread)
        target_link_options(${JYPPX_TARGET} PRIVATE -fsanitize=thread)
    endforeach()
endif()

# ============================================
# Coverage (Optional)
# ============================================
if(JYPPX_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(JYPPX_TARGET IN LISTS JYPPX_COMPILED_TARGETS)
        target_compile_options(${JYPPX_TARGET} PRIVATE --coverage)
        target_link_options(${JYPPX_TARGET} PRIVATE --coverage)
    endforeach()
endif()
