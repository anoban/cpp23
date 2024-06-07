// clang .\isitreamit.cpp -Wall -Wextra -pedantic -O3 -std=c++20

#include <algorithm>
#include <iostream>
#include <vector>

template<typename T> requires std::is_arithmetic_v<T> using wostream_iterator = std::ostream_iterator<T, wchar_t>;
template<typename T> requires std::is_arithmetic_v<T> using wistream_iterator = std::istream_iterator<T, wchar_t>;

auto wmain() {
    std::vector<float> values {};
    std::copy(wistream_iterator<float> { std::wcin }, wistream_iterator<float> {}, std::back_inserter(values));
    std::ranges::copy(values, wostream_iterator<float> { std::wcout, L"\n" });
    return EXIT_SUCCESS;
}
