#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <type_traits>

// implement a templated function that can return the sizes of a list of types
template<typename T, typename... TList> [[nodiscard]] std::initializer_list<unsigned> consteval sizesof() noexcept {
    return std::initializer_list<unsigned> { sizeof(T), sizesof<TList...>() };
}

template<typename T> [[nodiscard]] unsigned consteval sizesof() noexcept { return sizeof(T); }

auto wmain() -> int {
    constexpr auto sizes = ::sizesof<char, short, unsigned, long, long long, float, double, long double>();

    return EXIT_SUCCESS;
}
