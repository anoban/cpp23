#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

auto main() -> int {
    std::vector<double> array {};
    array.resize(10'000'000);

    std::mt19937_64 reng { std::random_device {}() };
    std::generate(array.begin(), array.end(), reng);
    const auto sum = std::accumulate(array.cbegin(), array.cend(), 0.0);

    std::cout << "sum is " << sum << std::endl;

    return EXIT_SUCCESS;
}
