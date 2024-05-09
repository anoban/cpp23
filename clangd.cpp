#include <cstdlib>
#include <type_traits>

template<typename T> struct is_unsigned;

template<> struct is_unsigned<unsigned char> {
        using type = unsigned char;
        static constexpr bool value { true };
};

template<> struct is_unsigned<unsigned short> {
        using type = unsigned short;
        static constexpr bool value { true };
};

template<> struct is_unsigned<unsigned int> {
        using type = unsigned int;
        static constexpr bool value { true };
};

template<> struct is_unsigned<unsigned long> {
        using type = unsigned long;
        static constexpr bool value { true };
};

template<> struct is_unsigned<unsigned long long> {
        using type = unsigned long long;
        static constexpr bool value { true };
};

template<bool, typename T> struct predicate_if;

template<typename T> struct predicate_if<true, T> {
        using type = T;
        static constexpr bool value { true };
        constexpr bool        operator()() const noexcept { return value; }
};

template<typename T, typename = typename ::predicate_if<::is_unsigned<T>::value, T>::type>
constexpr typename std::enable_if<!std::is_signed<T>::value, T>::type usum(const T x, const T y) noexcept {
    return x + y;
}

auto main() -> int {
    constexpr auto x { ::usum(12U, 645U) };

    constexpr auto s { ::usum(12U, 0.3546321) };
    return EXIT_SUCCESS;
}
