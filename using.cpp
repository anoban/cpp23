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

template<> struct is_unsigned<unsigned> {
        using type = unsigned;
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

template<typename T> static constexpr bool is_unsigned_v = ::is_unsigned<T>::value;

template<typename T> using is_unsigned_t                 = typename ::is_unsigned<typename std::remove_cv<T>::type>::type;

template<typename T> struct name {
        using value_type = std::enable_if<std::is_arithmetic<T>::value, T>::type;
        constexpr value_type operator()() const noexcept { return value; }

    private:
        value_type value {};
};

auto wmain() -> int {
    static_assert(std::is_same_v<decltype(::is_unsigned_v<unsigned short>), const bool>);
    static_assert(std::is_same<::is_unsigned_t<unsigned short>, unsigned short>::value);
    static_assert(::is_unsigned_v<float>);
    static_assert(::is_unsigned_v<long>);
    return EXIT_SUCCESS;
}
