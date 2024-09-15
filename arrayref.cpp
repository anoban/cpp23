#include <cctype>
#include <cstdio>
#include <cstdlib>

// signature of an array reference is T (&identifier)[length]

static __declspec(noinline) double __stdcall cube(const float& x) noexcept { return x * x * x; }
static __declspec(noinline) double __stdcall quad(const float& x) noexcept { return x * x * x * x; }

template<class T, size_t len> static __declspec(noinline) void __stdcall quad(T (&array)[len]) noexcept {
    for (T& e : array) e *= (e * e * e);
}

template<size_t __len> static constexpr inline void capitalize(_Inout_ char (&string)[__len]) noexcept {
    for (auto& c : string) c = static_cast<char>(::toupper(c));
}

template<class T, size_t __len>
static constexpr inline void square(_Inout_ T (&array)[__len]) noexcept requires requires(T& x) { x *= x; } {
    for (size_t i = 0; i < __len; ++i) array[i] *= array[i];
}

auto wmain() -> int {
    double (*const ptrcube)(const float&) noexcept = cube;
    //  ptrcube = quad;

    unsigned array[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    for (const auto& e : array) ::wprintf_s(L"%6u\n", e);
    ::_putws(L"\n");

    square(array);
    for (const auto& e : array) ::wprintf_s(L"%6u\n", e);

    ::wprintf_s(L"%lf\n", (*ptrcube)(3.000));

    char adele[] { "Let the sky fall.............. when it crumbles...." };
    ::capitalize(adele);
    ::wprintf_s(L"%S\n", adele);

    quad(array);
    for (const auto& e : array) ::wprintf_s(L"%6u\n", e);

    return EXIT_SUCCESS;
}
