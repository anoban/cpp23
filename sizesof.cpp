#include <cstdio>
#include <cstdlib>
#include <type_traits>

template<class C, typename T, typename... TList, unsigned size, unsigned... sizes> requires std::is_arithmetic_v<T> && std::is_class_v<C>
[[nodiscard]] consteval bool sizes_of() noexcept {
    if constexpr (!sizeof...(TList)) return true; // when the parameter pack is exhausted,
    return sizeof(C<T>) == size && sizes_of<C, TList..., sizes...>();
}

auto wmain() -> int {
    //
    return EXIT_SUCCESS;
}
