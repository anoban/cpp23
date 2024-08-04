#pragma once
#ifndef __DRY_BEANS_HPP__
    #define __DRY_BEANS_HPP__

// clang-format off
    #define _AMD64_ // architecture
    #define WIN32_LEAN_AND_MEAN
    #define WIN32_EXTRA_MEAN
    #include <WinDef.h>
    #include <WinBase.h>
    #include <errhandlingapi.h>
    #include <fileapi.h>
    #include <handleapi.h>
// clang-format on

    #include <algorithm>
    #include <cassert>
    #include <cstdio>
    #include <ranges>
    #include <string>
    #include <type_traits>
    #include <vector>

    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>

template<template<class> class _TypeTrait, class... _TList> static consteval bool all_of_trait_v() noexcept {
    return (_TypeTrait<_TList>::value && ...); // using fold expressions
}

template<template<class> class _TypeTrait, class T, class... _TList> static consteval bool any_of_trait_v() noexcept {
    if constexpr (sizeof...(_TList) == 0)
        return _TypeTrait<T>::value;
    else
        return (_TypeTrait<T>::value || ::any_of_trait_v<_TypeTrait, _TList...>());
}

static_assert(::all_of_trait_v<std::is_floating_point, float, double, long double>());
static_assert(!::all_of_trait_v<std::is_floating_point, float, double, long double, unsigned>());
static_assert(::all_of_trait_v<std::is_integral, char, short, long, unsigned, long long>());
static_assert(!::all_of_trait_v<std::is_integral, char, short, long, unsigned, long long, std::string>());

static_assert(::any_of_trait_v<std::is_floating_point, float, double, long double>());
static_assert(!::any_of_trait_v<std::is_floating_point, float*, const double&, volatile long double&&>());
static_assert(!::any_of_trait_v<std::is_floating_point, char&, short*, int, long&&, unsigned, long long>());
static_assert(::any_of_trait_v<std::is_integral, char, short, long, unsigned, long long, std::string, float, const double&&>());
static_assert(!::any_of_trait_v<std::is_integral, float, double, long double>());
static_assert(::any_of_trait_v<std::is_arithmetic, float, double, long double, char, short, int, long, unsigned long long>());

// yeeehawww :)

template<typename T> struct record final {
        T    area;
        T    perimeter;
        T    major_axis_length;
        T    minor_axis_length;
        T    aspect_ratio;
        T    eccentricity;
        T    convex_area;
        T    equiv_diameter;
        T    extent;
        T    solidity;
        T    roundness;
        T    compactness;
        T    shape_factor_1;
        T    shape_factor_2;
        T    shape_factor_3;
        T    shape_factor_4;
        char variety[10]; // max is 9 so :)
};

static_assert(::all_of_trait_v<std::is_standard_layout, record<float>, record<double>, record<long double>>()); // ;)

[[nodiscard]] static inline std::string __cdecl open(
    _In_ const wchar_t* const filename, _Inout_ unsigned long* rbytes
) noexcept(std::is_nothrow_constructible_v<std::string>) {
    *rbytes = 0;

    std::string    buffer {};
    LARGE_INTEGER  filesize {};
    const HANDLE64 hFile =
        ::CreateFileW(filename, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, nullptr);

    if (hFile == INVALID_HANDLE_VALUE) {
        ::fprintf_s(stderr, "Error %lu in CreateFileW\n", ::GetLastError()); // NOLINT(cppcoreguidelines-pro-type-vararg)
        return std::string {};
    }

    if (!::GetFileSizeEx(hFile, &filesize)) {
        ::fprintf_s(stderr, "Error %lu in GetFileSizeEx\n", ::GetLastError()); // NOLINT(cppcoreguidelines-pro-type-vararg)
        goto ERROR_EXIT;
    }

    buffer.resize(filesize.QuadPart + 1); // +1 for the null terminator

    if (!::ReadFile(hFile, buffer.data(), filesize.QuadPart, rbytes, nullptr)) {
        ::fprintf_s(stderr, "Error %lu in ReadFile\n", ::GetLastError()); // NOLINT(cppcoreguidelines-pro-type-vararg)
        goto ERROR_EXIT;
    }

    ::CloseHandle(hFile);
    return buffer;

ERROR_EXIT:
    ::CloseHandle(hFile);
    return std::string {}; // buffer will be destroyed
}

#endif // __DRY_BEANS_HPP__
