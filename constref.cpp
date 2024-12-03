#include <cstdio>
#include <cstdlib>

static __declspec(noinline) double __stdcall power(_In_ const double& b, _In_ const unsigned short& exp) noexcept {
    ::_putws(L"" __FUNCSIG__);
    if (!exp) return 1.00;
    if (exp == 1) return b;
    double result { b };
    for (unsigned i = 1; i < exp; ++i) result *= b;
    return result;
}

static __declspec(noinline) double __stdcall power(_In_ const float b, _In_ const unsigned short exp) noexcept {
    ::_putws(L"" __FUNCSIG__);
    if (!exp) return 1.00;
    if (exp == 1) return b;
    double result { b };
    for (unsigned i = 1; i < exp; ++i) result *= b;
    return result;
}

static __declspec(noinline) double __stdcall cube(_In_ const double& value) noexcept { return value * value * value; }

auto wmain() -> int {
    const auto four { ::power(2.000, 2) }; // the compiler always prefers the const reference overload!
    ::wprintf_s(L"%.5lf\n", four);

    constexpr double   three { 3.00000 };
    constexpr unsigned five { 5 };

    const auto         what { ::power(three, five) };
    ::wprintf_s(L"%.5lf\n", what);

    const auto nine { ::cube(three) };
    const auto sixty_four { ::cube(4.0000) };
    // a function taking const T& should be able to bind both lvalues and rvalues!
    // hence the requirement an identifier of type const T& should also be able to bind values of type T&& (rvalues of type T)

    return EXIT_SUCCESS;
}
