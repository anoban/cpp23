#ifdef _WIN32
    #define __STDC_WANT_SECURE_LIB__ 1
    #define WIN32_LEAN_AND_MEAN
    #define WIN32_EXTRA_MEAN
#endif

#include <iostream>
#include <print>
// to dump all defines with clang, clang -dM -E
// even with -std=c++23, only clang supports both std::print and std::format
// MSVC supports std::format with /std:c++20 but requires /std:c++latest for std::print, std::println

#if defined(__cplusplus)

// compiler specifics

    #if defined(__GNUC__) && defined(__MINGW32__) // MSYS2 g++  g++ cpp23.cpp -Wall -Wextra -O3 -std=c++20
        #if !defined(__cpp_lib_print)
            #error "Support for std::print unavailable in g++!"
        #endif
        #ifndef __cpp_lib_format
            #error "Support for std::format unavailable in g++!"
        #endif
    #endif

    #if defined(__llvm__) && defined(__clang__) // clang++ clang .\cpp23.cpp -Wall -O3 -std=c++23 -Wextra -pedantic
        #if !defined(__cpp_lib_print)
            #error "Support for std::print unavailable in clang++!"
        #endif
        #ifndef __cpp_lib_format
            #error "Support for std::format unavailable in clang++!"
        #endif
    #endif

    // clang uses Microsoft's STL, so clang will also define _MSC_VER
    // checks specific for cl.exe - cl .\cpp23.cpp /Wall /std:c++latest /EHac /Ot /O2 /wd4711 /wd4710
    #if !defined(__clang__) && defined(_MSC_VER) && defined(_MSVC_LANG) && defined(_MSC_FULL_VER) && !defined(__cpp_lib_print)
        #error "Support for std::print unavailable in MSVC++!"
    #endif

    #if !defined(__clang__) && defined(_MSC_VER) && defined(_MSVC_LANG) && defined(_MSC_FULL_VER) && !defined(__cpp_lib_format)
        #error "Support for std::format unavailable in MSVC++!"
    #endif

#else // !defined(__cplusplus)
    #error "A C++ compiler needed!"
#endif

int main() {
    std::wcout << L"C++ standard version " << __cplusplus << L'\n';
    std::println("C++ standard version {}", __cplusplus); // looks like std::println doesn't yet have overloads for wchar_t
#ifdef _MSVC_LANG
    std::print("_MSVC_LANG {}\n", _MSVC_LANG);            // even with /std:latest MSVC uses 202004L for _MSVC_LANG
#endif
    // with /std:c++latest, __cplusplus seems to be defined 199711L in MSVC
    // with clang++ it is 202302L
    return EXIT_SUCCESS;
}
