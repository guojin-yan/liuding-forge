# Compiler Options for JYPPX Training Framework
#
# Per-configuration flags use generator expressions so they work with both
# single-config (Ninja/Make) and multi-config (Visual Studio) generators.

# ============================================
# Warning Flags
# ============================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(jyppx_training_framework PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O2>
        $<$<CONFIG:Debug>:-O0;-g>
    )
elseif(MSVC)
    target_compile_options(jyppx_training_framework PRIVATE
        /W4
        /wd4100  # Unused parameter
        # Optimization levels are already provided per-config by
        # CMAKE_CXX_FLAGS_<CONFIG> (/O2 for Release, /Od for Debug).
    )
endif()

# ============================================
# Debug Definitions
# ============================================
target_compile_definitions(jyppx_training_framework PRIVATE
    $<$<CONFIG:Debug>:JYPPX_DEBUG>
)

# ============================================
# Sanitizers (Optional)
# ============================================
option(JYPPX_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(JYPPX_ENABLE_TSAN "Enable Thread Sanitizer" OFF)

if(JYPPX_ENABLE_ASAN AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(jyppx_training_framework PRIVATE -fsanitize=address)
    target_link_options(jyppx_training_framework PRIVATE -fsanitize=address)
endif()

if(JYPPX_ENABLE_TSAN AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(jyppx_training_framework PRIVATE -fsanitize=thread)
    target_link_options(jyppx_training_framework PRIVATE -fsanitize=thread)
endif()

# ============================================
# Coverage (Optional)
# ============================================
if(JYPPX_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(jyppx_training_framework PRIVATE --coverage)
    target_link_options(jyppx_training_framework PRIVATE --coverage)
endif()
