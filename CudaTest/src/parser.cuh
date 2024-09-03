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
    #include <charconv>
    #include <concepts>
    #include <iomanip>
    #include <iostream>
    #include <numeric>
    #include <ranges>
    #include <string>
    #include <string_view>
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

template<typename T> class record final {
    public:
        using value_type = T;
        unsigned long long area;
        T                  perimeter;
        T                  major_axis_length;
        T                  minor_axis_length;
        T                  aspect_ratio;
        T                  eccentricity;
        T                  convex_area;
        T                  equiv_diameter;
        T                  extent;
        T                  solidity;
        T                  roundness;
        T                  compactness;
        T                  shape_factor_1;
        T                  shape_factor_2;
        T                  shape_factor_3;
        T                  shape_factor_4;
        char               variety[10]; // max is 9 so :)

        template<typename U> requires std::floating_point<U> __host__ __device__ bool operator==(const record<U>& other) const noexcept {
            return !::memcmp(variety, other.variety, __crt_countof(variety));
        }

        template<typename U> requires std::floating_point<U> __host__ __device__ bool operator!=(const record<U>& other) const noexcept {
            return ::memcmp(variety, other.variety, __crt_countof(variety));
        }

        template<typename U> requires std::floating_point<U>
        __host__ __device__ record<long double> operator+(const record<U>& other) const noexcept {
            return {
                area + other.area,
                perimeter + other.perimeter,
                major_axis_length + other.major_axis_length,
                minor_axis_length + other.minor_axis_length,
                aspect_ratio + other.aspect_ratio,
                eccentricity + other.eccentricity,
                convex_area + other.convex_area,
                equiv_diameter + other.equiv_diameter,
                extent + other.extent,
                solidity + other.solidity,
                roundness + other.roundness,
                compactness + other.compactness,
                shape_factor_1 + other.shape_factor_1,
                shape_factor_2 + other.shape_factor_2,
                shape_factor_3 + other.shape_factor_3,
                shape_factor_4 + other.shape_factor_4,
            };
            // ignoring the member `variety`
        }

        template<typename U> requires std::floating_point<U> __host__ __device__ record<T>& operator+=(const record<U>& other) noexcept {
            area              += other.area;
            perimeter         += other.perimeter;
            major_axis_length += other.major_axis_length;
            minor_axis_length += other.minor_axis_length;
            aspect_ratio      += other.aspect_ratio;
            eccentricity      += other.eccentricity;
            convex_area       += other.convex_area;
            equiv_diameter    += other.equiv_diameter;
            extent            += other.extent;
            solidity          += other.solidity;
            roundness         += other.roundness;
            compactness       += other.compactness;
            shape_factor_1    += other.shape_factor_1;
            shape_factor_2    += other.shape_factor_2;
            shape_factor_3    += other.shape_factor_3;
            shape_factor_4    += other.shape_factor_4;
            // ignoring the member `variety`
            return *this;
        }

        friend std::ostream& operator<<(_Inout_ std::ostream& ostream, _In_ const record& rcrd) noexcept {
            ostream << "Area " << rcrd.area << '\n'
                    << "Perimeter " << rcrd.perimeter << '\n'
                    << "MajorAxisLength " << rcrd.major_axis_length << '\n'
                    << "MinorAxisLength " << rcrd.minor_axis_length << '\n'
                    << "AspectRation " << rcrd.aspect_ratio << '\n'
                    << "Eccentricity " << rcrd.eccentricity << '\n'
                    << "ConvexArea " << rcrd.convex_area << '\n'
                    << "EquivDiameter " << rcrd.equiv_diameter << '\n'
                    << "Extent " << rcrd.extent << '\n'
                    << "Solidity " << rcrd.solidity << '\n'
                    << "roundness " << rcrd.roundness << '\n'
                    << "Compactness " << rcrd.compactness << '\n'
                    << "ShapeFactor1 " << rcrd.shape_factor_1 << '\n'
                    << "ShapeFactor2 " << rcrd.shape_factor_2 << '\n'
                    << "ShapeFactor3 " << rcrd.shape_factor_3 << '\n'
                    << "ShapeFactor4 " << rcrd.shape_factor_4 << '\n';
            return ostream;
        }
};

static_assert(::all_of_trait_v<std::is_standard_layout, record<float>, record<double>, record<long double>>()); // ;)

[[nodiscard]] static inline std::string __cdecl open(_In_ const wchar_t* const filename, _Inout_ unsigned long* rbytes)
    noexcept(std::is_nothrow_constructible_v<std::string>) {
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
