// clang .\bindcompose.cpp -Wall -O3 -std=c++20 -Wextra -pedantic

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>

int main() {
    // count of numbers < (RAND_MAX / 2) and > 2000
    auto a { std::array<int, 1000> {} };
    for (std::array<int, 1000>::iterator it = a.begin(), end = a.end(); it != end; ++it) *it = rand();

    size_t qualified {};
    for (const auto& e : a) qualified += (e < (RAND_MAX / 2)) && (e > 1000);

    // this is cool as fuck but terribly unreadable
    // this composition of two separate predicates into a final one using a logical and operation is inefficient too
    // all this just for qualified += (e < (RAND_MAX / 2)) && (e > 1000);

    constexpr auto predicate_lt       = std::bind(std::less<int> {}, std::placeholders::_1, RAND_MAX / 2);
    constexpr auto predicate_gt       = std::bind(std::greater<int> {}, std::placeholders::_1, 1000);
    constexpr auto predicate_composed = std::bind(std::logical_and<bool> {}, predicate_gt, predicate_lt);

    const auto     qual               = std::count_if(a.begin(), a.end(), predicate_composed);
    std::wcout << qualified << L' ' << qual << std::endl;
    return EXIT_SUCCESS;
}
