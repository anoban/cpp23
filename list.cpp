// clang .\list.cpp -Wall -Wextra -pedantic -O3 -std=c++20 -o list.exe

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <ranges>
#include <type_traits>

template<typename T> [[nodiscard]] constexpr size_t distance(const T start, const T end) noexcept {
    std::iterator_traits<T>::value_type;
    return (static_cast<char*>(end) - static_cast<char*>(start)) / sizeof(T);
}

int wmain() {
    std::list<double> list;
    for (const auto& d : std::ranges::views::iota(0, 100)) list.push_back(static_cast<double>(d));

    std::wcout << list.front() << L'\n';
    std::wcout << list.back() << L'\n';

    std::wcout << std::distance(list.cbegin(), list.cend()) << L" doubles! \n";
    // linked lists do not provide random access iterators yikes!
    // std::wcout << ::distance(list.cbegin(), list.cend()) << L" doubles! \n";
    std::array<float, 100> array;
    std::wcout << ::distance(array.cbegin(), array.cend()) << L" doubles! \n";

    return EXIT_SUCCESS;
}
