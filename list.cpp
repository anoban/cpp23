// clang .\list.cpp -Wall -Wextra -pedantic -O3 -std=c++20 -o list.exe

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <ranges>
#include <type_traits>
#include <vector>

// assumes the inputs are random access iterators and expects class `const_iterator` to have a typedef named `value_type`
template<typename const_iterator>
[[nodiscard]] constexpr ptrdiff_t distance(const const_iterator first, const const_iterator last) noexcept {
    using value_type = const_iterator::value_type; // okay
    // return end._Unwrapped() - begin._Unwrapped();    // okay too :)
    static_assert(!std::is_pointer<value_type>::value);
    return (reinterpret_cast<const char*>(last._Unwrapped()) - reinterpret_cast<const char*>(first._Unwrapped())) /
           static_cast<ptrdiff_t>(sizeof(value_type));
}

int wmain() {
    std::list<double> list;
    for (const auto& d : std::ranges::views::iota(0, 100)) list.push_back(static_cast<double>(d));

    std::wcout << list.front() << L'\n';
    std::wcout << list.back() << L'\n';

    std::wcout << std::distance(list.cbegin(), list.cend()) << L" doubles! \n";
    // linked lists do not provide random access iterators

    // std::wcout << ::distance(list.cbegin(), list.cend()) << L" doubles! \n";
    std::array<float, 100> array {};
    std::vector<short>     vector(10000);

    std::wcout << ::distance(array.cbegin(), array.cend()) << L" floats! \n";
    std::wcout << std::distance(array.cbegin(), array.cend()) << L" floats! \n";

    std::wcout << ::distance(vector.begin(), vector.end()) << L" shorts! \n";
    std::wcout << std::distance(vector.begin(), vector.end()) << L" shorts! \n";

    return EXIT_SUCCESS;
}
