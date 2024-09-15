#include <cstdio>
#include <cstdlib>
#include <cstring>

// signature of an array reference is T (&ident)[length]

static __declspec(noinline) double __stdcall cube(const float& x) noexcept { return x * x * x; }
static __declspec(noinline) double __stdcall quad(const float& x) noexcept { return x * x * x * x; }

template<size_t __len> static constexpr inline void capitalize(_Inout_ char (&string)[__len]) noexcept { }

template<class T, size_t __len> static constexpr inline void square(_Inout_ T (&array)[__len]) noexcept {
    for (size_t i = 0; i < __len; ++i) array[i] = array[i] * array[i];
}

auto wmain() -> int {
    double (*const ptrcube)(const float&) noexcept = cube;
    //  ptrcube                                        = quad;

    unsigned array[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    for (const auto& e : array) ::wprintf_s(L"%6u\n", e);

    square(array);
    for (const auto& e : array) ::wprintf_s(L"%6u\n", e);

    ::wprintf_s(L"%lf\n", (*ptrcube)(3.000));

    return EXIT_SUCCESS;
}
