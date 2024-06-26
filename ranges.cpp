#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

int main() {
    [[maybe_unused]] constexpr auto value { 54I16 };

    std::vector<int32_t> randoms {};
    randoms.resize(100);
    std::iota(randoms.begin(), randoms.end(), 0);

    auto odds { randoms | std::ranges::views::filter([](const int32_t& number) { return !(number % 2); }) |
                std::ranges::views::transform([](int32_t& number) { return number * number; }) };

    auto evens { randoms | std::ranges::views::filter([](const int32_t& number) { return number % 2; }) |
                 std::ranges::views::transform([](int32_t& number) { return number * number; }) };

    for (const auto& oddelem : odds) std::wcout << oddelem << L' ';
    std::wcout << L'\n';
    std::wcout << L'\n';

    for (const auto& evenelem : evens) std::wcout << evenelem << L' ';
    std::wcout << L'\n';
    std::wcout << L'\n';

    for (const auto& i : std::ranges::views::iota(10)) {
        std::wcout << i << L' ';
        if (i > 25) break;
    }
    std::wcout << L'\n';
    std::wcout << L'\n';

    for (const auto& i : std::ranges::views::iota(100, 115)) std::wcout << i << L' ';

    return EXIT_SUCCESS;
}
