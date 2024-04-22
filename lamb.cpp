// clang .\lamb.cpp -O3 -std=c++20 -Wall -Wextra -pedantic

#include <algorithm>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

// https://en.cppreference.com/w/cpp/types/enable_if

template<
    typename scalar_t,
    typename = std::enable_if<std::is_floating_point<scalar_t>::value>::type> // leveraging enable_if as a template type parameter
static std::wostream& operator<<(std::wostream& wostr, const std::vector<scalar_t>& collection) {
    wostr << L"[ ";
    for (typename std::vector<scalar_t>::const_iterator it = collection.cbegin(), end = collection.cend(); it != end; ++it)
        wostr << *it << L", ";
    wostr << L"\b\b ]";
    return wostr;
}

#define SIZE 400LLU

int main() {
    std::vector<float> randoms(SIZE);
    std::random_device rdevice {};
    std::mt19937_64    rndengine { rdevice() };

    std::ranges::fill(randoms, std::generate_canonical<float, 24>(rndengine));
    std::wcout << randoms;

    return EXIT_SUCCESS;
}
