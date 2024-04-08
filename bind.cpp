#include <cstdio>
#include <functional>
#include <numbers>

static void                func(const float& a) noexcept { ::wprintf_s(L"a = %f\n", a); }

static void                func2args(short a, long b) noexcept { ::wprintf_s(L"a = %hd, b = %ld\n", a, b); }

[[nodiscard]] static float func3args(float a, float b, float c) noexcept {
    ::wprintf_s(L"a = %f, b = %f, c = %f\n", a, b, c);
    return a + b + c;
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

    return EXIT_SUCCESS;
}
