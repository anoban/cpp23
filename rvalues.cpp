// clang .\rvalues.cpp -Wall -O3 -std=c++23 -Wextra -pedantic

#include <cstdlib>
#include <iostream>
#include <numbers>
#include <type_traits>

template<typename T, typename = std::enable_if<std::is_scalar<T>::value, T>::type> class pair {
    private:
        T first;
        T second;

    public:
        constexpr pair() noexcept : first(), second() { }

        constexpr explicit pair(const T& _val) noexcept : first(_val), second(_val) { }

        constexpr explicit pair(const T& _val0, const T& _val1) noexcept : first(_val0), second(_val1) { }

        constexpr ~pair() = default;

        // copy ctor, move ctor, copy assignment and move assignment operators cannot be templates!
        // who knew!
        constexpr pair(const pair& other) noexcept : first(other.first), second(other.second) { }

        constexpr pair(pair&& other) noexcept : first(other.first), second(other.second) { }

        constexpr pair& operator=(const pair& other) noexcept {
            if (&other == this) return *this;
            first  = other.first;
            second = other.second;
            return *this;
        }

        constexpr pair& operator=(pair&& other) noexcept {
            if (&other == this) return *this;
            first  = other.first;
            second = other.second;
            return *this;
        }

        template<typename U> constexpr pair<typename std::enable_if<std::is_scalar<U>::value, U>::type> to() const noexcept {
            return pair<U> { static_cast<U>(first), static_cast<U>(second) };
        }

        template<typename char_t> friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const pair& object) {
            // using function style casts
            ostream << char_t('{') << char_t(' ') << object.first << char_t(',') << char_t(' ') << object.second << char_t(' ')
                    << char_t('}');
            return ostream;
        }
};

int wmain() {
    constexpr ::pair<float> a { std::numbers::pi_v<float> };
    std::wcout << a << std::endl;

    constexpr auto x { ::pair<short> {} };
    std::wcout << x << std::endl;

    constexpr auto y {
        ::pair<double> { 12.086, 6543.0974 }
    };
    std::wcout << y << std::endl;

    constexpr auto z { y };
    ::pair<float>  q {};
    q = a;
    std::wcout << q << std::endl;

    constexpr auto to { y.to<float>() };
    std::wcout << to << std::endl;

    q = q;

    return EXIT_SUCCESS;
}
