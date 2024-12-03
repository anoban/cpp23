#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numbers>

template<typename scalar_t, typename char_t, size_t size>
requires std::is_arithmetic<scalar_t>::value && (std::is_same_v<char_t, char> || std::is_same_v<char_t, wchar_t>)
static std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostr, const std::array<scalar_t, size>& array) {
    ostr << L"[ ";
    for (std::array<scalar_t, size>::const_iterator it = array.cbegin(), e = array.cend(); it != e; ++it)
        ostr << std::setw(3) << *it << L", ";
    ostr << L"]\n" << std::setw(0);
    return ostr;
}

static void                func(const float& a) noexcept { ::wprintf_s(L"a = %f\n", a); }

static void                func2args(short a, long b) noexcept { ::wprintf_s(L"a = %hd, b = %ld\n", a, b); }

[[nodiscard]] static float func3args(float a, float b, float c) noexcept {
    ::wprintf_s(L"a = %f, b = %f, c = %f\n", a, b, c);
    return a + b + c;
}

// elementwise multiplication
template<typename scalar_t, size_t length> requires std::is_arithmetic_v<scalar_t> static constexpr std::array<double, length> func4arrays(
    const std::array<scalar_t, length>& a, const std::array<scalar_t, length>& b
) noexcept {
    std::array<double, length> res {};
    std::transform(a.cbegin(), a.cend(), b.cbegin(), res.data(), std::multiplies {});
    return res;
}

static void func5args(int a, int b, int c, const int* const pd, int e) noexcept {
    ::wprintf_s(L"a = %d, b = %d, c = %d, d = %d, e = %d\n", a, b, c, *pd, e);
    return;
}

auto main() -> int {
    constexpr auto  x { 43 };
    constexpr float y { 8.764571 };

    // to make a std::bind object constant, all it's non-value type arguments needs to be constants
    const auto      f1 = std::bind(func, y);
    f1();

    const auto f2 = std::bind(func2args, 34I16, std::placeholders::_1);
    f2(2434L);

    // placeholders can be exploited to reorder the sequence of arguments passed to the original function.
    const auto f3 = std::bind(func3args, std::placeholders::_2, std::placeholders::_1, std::numbers::pi_v<float> * 3);
    f3(std::numbers::pi * 2 /* 2nd arg to func3args */, std::numbers::pi /* 1st arg to func3args */);

    const auto f3r = std::bind(func3args, std::placeholders::_3, std::placeholders::_2, std::placeholders::_1);
    // f3r will pass the arguments it received in reverse order to the bound function
    f3r(std::numbers::e, std::numbers::e * 2, std::numbers::e * 3);

    // placeholders in std::bind* functions specify the arguments passed to the bound function at call site
    const auto f5 = std::bind(
        func5args,
        std::placeholders::_1, // means that the first argument to default_1 one will be treated as a (1st argument to func5args)
        100,                   // second argument to func5args()
        std::placeholders::_2, // third argument to func5args()
        &x,                    // fourth argument - const int* const
        std::placeholders::_3  // fifth argument to func5args()
    );
    f5(23, 19, 444);

    // std::bind preserves noexcept() qualifiers too
    static_assert(noexcept(f5(22, 45, 99)));

    constexpr std::array<int, 10> array_0 { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    constexpr std::array<int, 10> array_1 { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    std::wcout << array_0;
    std::wcout << array_1;

    constexpr auto f4 = std::bind_front(func4arrays<int, 10>, array_0);
    std::wcout << f4(array_1);

    const auto f4b = std::bind_back(func4arrays<int, 10>, array_1); // bind_back needs C++23
    std::wcout << f4b(array_0);

    return EXIT_SUCCESS;
}
