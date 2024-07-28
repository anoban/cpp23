#include <array>
#include <cstdlib>
#include <type_traits>

// woohoo motherfuckers :)
template<typename... TList> [[nodiscard]] static auto consteval sizesof() noexcept -> std::array<size_t, sizeof...(TList)> {
    return { sizeof(TList)... };
}

template<class T> static auto consteval sum_sizes() noexcept -> size_t { return sizeof(T); }

// without the constraint requires(sizeof...(TList) != 0), we'll get an overload ambiguity error
// with sizesof<T = T>() and sizesof<T = T, TList = <>> when only one element is left in the type list!
template<class T, class... TList> requires(sizeof...(TList) != 0) static consteval auto sum_sizes() noexcept -> size_t {
    return sizeof(T) + ::sum_sizes<TList...>();
}

template<unsigned n> [[nodiscard]] consteval unsigned sum() noexcept { return n; }

// partial specializations cannot be implemented for functions
// template<unsigned n, unsigned... numbers, bool is_pack_empty = sizeof...(numbers) != 0> [[nodiscard]] consteval unsigned sum() noexcept {
//      return n + ::sum<numbers...>();
// }

// template<unsigned n, unsigned... numbers> [[nodiscard]] consteval unsigned sum<n, numbers..., false>() noexcept = delete;

// when std::enable_if_t<sizeof...(numbers) != 0, bool> is not well formed, fall back to the scalar overload
template<unsigned n, unsigned... numbers, class = std::enable_if_t<sizeof...(numbers) != 0, bool>>
[[nodiscard]] consteval unsigned sum() noexcept {
    return n + ::sum<numbers...>();
}

auto wmain() -> int {
    [[maybe_unused]] constexpr auto sizes_sum = ::sum_sizes<char, short, unsigned, long, long long, float, double, long double>();
    [[maybe_unused]] constexpr auto size_list = ::sizesof<char, short, unsigned, long, long long, float, double, long double>();
    [[maybe_unused]] constexpr auto total     = ::sum<81, 55, 29, 17, 68, 71, 29, 66, 35, 97>(); // 548

    return EXIT_SUCCESS;
}
