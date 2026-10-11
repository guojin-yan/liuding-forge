#ifndef JYPPX_VERSION_HPP
#define JYPPX_VERSION_HPP

#define JYPPX_VERSION_MAJOR 0
#define JYPPX_VERSION_MINOR 3
#define JYPPX_VERSION_PATCH 0

#define JYPPX_VERSION "0.3.0"

namespace jyppx {

/**
 * @brief Get JYPPX version string
 * @return Version string (e.g., "0.1.0")
 */
inline const char* version() {
    return JYPPX_VERSION;
}

/**
 * @brief Get major version number
 */
inline int version_major() {
    return JYPPX_VERSION_MAJOR;
}

/**
 * @brief Get minor version number
 */
inline int version_minor() {
    return JYPPX_VERSION_MINOR;
}

/**
 * @brief Get patch version number
 */
inline int version_patch() {
    return JYPPX_VERSION_PATCH;
}

} // namespace jyppx

#endif // JYPPX_VERSION_HPP
