// clang .\lam.cpp -Wall -Wextra -O3 -std=c++23 -pedantic

#include <algorithm>
#include <functional>
#include <iostream>
#include <numbers>
#include <random>
#include <ranges>
#include <vector>

constexpr auto bare_minimum = [] { };

constexpr auto give3        = [] { return 3; };

constexpr auto add1         = [](auto x) -> decltype(x) { return x + 1; };

constexpr auto add10        = [](auto x) consteval noexcept(noexcept(x + 10)) -> decltype(x) { return x + 10; };

constexpr auto sum { ::add10(12ui32) };

constexpr auto increment = [](auto& x) constexpr noexcept -> void { x = add1(x); };

constexpr long long threshold { 2000 };

int wmain() {
    { // scope
        constexpr auto pi { std::numbers::pi_v<float> };

        auto area = [&pi](auto radius) constexpr noexcept -> decltype(std::numbers::pi_v<double>) { return 2.000 * pi * radius; };
        auto ar { area(22) };
        std::wcout << ar << std::endl;
    }

    std::vector<int32_t> randoms(100'000);
    auto                 rdev { std::random_device {} };
    auto                 rengine { std::mt19937_64 { rdev.operator()() } };
    std::ranges::generate(randoms, rengine);

    const auto lt2000 { std::ranges::count_if(randoms, std::bind(std::less<int32_t> {}, std::placeholders::_1, threshold)) };
    size_t     count {};
    for (const auto& e : std::ranges::views::all(randoms)) count += e < threshold;

    std::wcout << lt2000 << L' ' << count << std::endl;

    // get the count of numbers between 100 and 1000
    const auto num = std::ranges::count_if(
        randoms,
        std::bind(
            std::logical_and<bool> {},
            std::bind(std::less<int32_t> {}, std::placeholders::_1, INT32_MAX / 2),
            std::bind(std::greater<int32_t> {}, std::placeholders::_1, INT32_MIN / 2)
        )
    );

    // wth?
    std::wcout << L"max is " << std::ranges::max(randoms) << L", min is " << *std::min(randoms.cbegin(), randoms.cend()) << std::endl;
    std::wcout << std::numeric_limits<int>::max() << L' ' << std::numeric_limits<int>::min() << std::endl;

    count = 0;
    for (const auto& e : randoms) count += ((e < (INT32_MAX / 2)) && (e > (INT32_MIN / 2)));
    std::wcout << num << L' ' << count << std::endl;

    return EXIT_SUCCESS;
}
