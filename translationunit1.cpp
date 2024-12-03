#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// templated functions need to be declared and defined in the same TU
// template<typename T> requires std::is_arithmetic_v<T> [[nodiscard]] double sum(const std::vector<T>& collection) noexcept;

[[nodiscard]] double sum(const std::vector<int>& collection) noexcept; // defined in translationunit0.cpp

int                  main() {
    std::vector<int>   randoms(1000);
    std::random_device seeder {};
    std::knuth_b       rengine { seeder() };

    std::generate(randoms.begin(), randoms.end(), rengine);
    const auto sum { ::sum(randoms) };
    std::wcout << sum << std::endl;

    return EXIT_SUCCESS;
}
