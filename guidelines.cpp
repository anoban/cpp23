#include <gsl>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

auto wmain() -> int {
    std::wcout << std::fixed << std::setprecision(20);

    std::vector<unsigned __int64> randoms(1000);
    std::vector<unsigned __int64> randoms_cpy(1000);
    std::mt19937_64               rengine { std::random_device {}() };
    std::ranges::generate(randoms, [&rengine]() noexcept -> unsigned __int64 { return rengine() % std::numeric_limits<unsigned>::max(); });

    gsl::copy(gsl::span(randoms), gsl::span(randoms_cpy));
    for (const auto& i : std::ranges::views::iota(0LLU, 1000LLU)) std::wcout << randoms.at(i) << L' ' << randoms_cpy.at(i) << L'\n';

    return EXIT_SUCCESS;
}
