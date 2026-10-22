# LibTorch 2.6+ may omit TORCH_CXX_FLAGS when using the compiler's default
# libstdc++ ABI. Bind the fingerprint to the effective consumer ABI and prove
# that an ABI-sensitive C10 symbol links with the selected binary package.
function(jyppx_detect_libtorch_abi result)
    if(MSVC)
        set(${result} "msvc-md-iterator0" PARENT_SCOPE)
        return()
    elseif(APPLE)
        set(${result} "libcxx" PARENT_SCOPE)
        return()
    endif()

    if(NOT TARGET c10)
        message(FATAL_ERROR "LibTorch ABI detection requires its imported c10 target")
    endif()
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${TORCH_INCLUDE_DIRS}")
    set(CMAKE_REQUIRED_LIBRARIES c10)
    # A retained build directory can switch Torch packages or compiler flags.
    # Do not reuse a successful probe from a different binary ABI.
    foreach(abi IN ITEMS 0 1)
        unset(JYPPX_LIBTORCH_ABI${abi}_LINKS CACHE)
        check_cxx_source_compiles("
#include <c10/util/Type.h>
#include <string>
#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != ${abi}
#error Consumer does not use libstdc++ ABI ${abi}
#endif
int main() {
    const std::string name = c10::demangle(\"i\");
    return name.empty() ? 1 : 0;
}
" JYPPX_LIBTORCH_ABI${abi}_LINKS)
    endforeach()
    if(JYPPX_LIBTORCH_ABI0_LINKS)
        set(${result} "cxx11-abi0" PARENT_SCOPE)
    elseif(JYPPX_LIBTORCH_ABI1_LINKS)
        set(${result} "cxx11-abi1" PARENT_SCOPE)
    else()
        message(FATAL_ERROR
            "The effective C++ ABI cannot link the selected LibTorch c10 library. "
            "Check the compiler, TORCH_CXX_FLAGS, and exact LibTorch package.")
    endif()
endfunction()
