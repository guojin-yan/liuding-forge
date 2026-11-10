// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef LIUDING_TEST_IMAGE_OUTPUT_WINDOWS_NATIVE_HOOKS_HPP
#define LIUDING_TEST_IMAGE_OUTPUT_WINDOWS_NATIVE_HOOKS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

extern "C" {
HANDLE WINAPI liuding_image_test_CreateFileW(LPCWSTR, DWORD, DWORD,
                                             LPSECURITY_ATTRIBUTES, DWORD,
                                             DWORD, HANDLE);
BOOL WINAPI liuding_image_test_GetVolumeInformationByHandleW(HANDLE, LPWSTR,
                                                             DWORD, LPDWORD,
                                                             LPDWORD, LPDWORD,
                                                             LPWSTR, DWORD);
BOOL WINAPI liuding_image_test_WriteFile(HANDLE, LPCVOID, DWORD, LPDWORD,
                                         LPOVERLAPPED);
BOOL WINAPI liuding_image_test_FlushFileBuffers(HANDLE);
BOOL WINAPI liuding_image_test_CreateHardLinkW(LPCWSTR, LPCWSTR,
                                               LPSECURITY_ATTRIBUTES);
BOOL WINAPI liuding_image_test_CloseHandle(HANDLE);
}

// Force-included only for the test's copy of the original sink source.
// The normal library calls the unmodified Windows APIs.
#define CreateFileW liuding_image_test_CreateFileW
#define GetVolumeInformationByHandleW \
  liuding_image_test_GetVolumeInformationByHandleW
#define WriteFile liuding_image_test_WriteFile
#define FlushFileBuffers liuding_image_test_FlushFileBuffers
#define CreateHardLinkW liuding_image_test_CreateHardLinkW
#define CloseHandle liuding_image_test_CloseHandle

#endif
