#include <numbers>
#include <type_traits>

template<bool, typename T> struct predicate_if; // handrolled std::enable_if alternative

// partial specialization where bool is true
template<typename T> struct predicate_if<true, T> {
        static constexpr bool value { true };
        using type = T;
};

// convenience wrapper for ::predicate_if<predicate, T>::value
template<bool predicate, typename T> constexpr bool predicate_if_v = predicate_if<predicate, T>::value;

// will work with floats and doubles not with long doubles
template<typename T, typename U>
consteval ::predicate_if<std::is_floating_point<T>::value && std::is_floating_point<U>::value, float>::type sum(
    const T x,
    const U y,
    const typename ::predicate_if<!std::is_same<typename std::remove_cv<T>::type, long double>::value, bool>::type = true
) throw() {
    return static_cast<float>(x + y);
}

template<typename T> consteval ::predicate_if<std::is_integral_v<T>, T>::type square(const T base) noexcept { return base * base; }

auto main() -> int {
    constexpr auto ldb { std::numbers::pi_v<long double> };
    constexpr auto x { ::sum(ldb, 1.0002F) };
    constexpr auto y { ::sum(65.7823, 12) };
    constexpr auto z { ::sum(65.7823, 12.567564F) };
    constexpr auto a { ::sum(65.7823, 12.567564) };

    constexpr auto must_be_true = ::predicate_if_v<std::is_scalar<float>::value, short>; // float is a scalar
    return 0;
}

