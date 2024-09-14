// clang .\rvalues.cpp -Wall -O3 -std=c++23 -Wextra -pedantic

#include <cstdlib>
#include <iostream>
#include <numbers>
#include <type_traits>

template<typename T, typename = std::enable_if<std::is_arithmetic<T>::value, T>::type> class pair final {
    private:
        T first;
        T second;
        template<typename, typename> friend class pair; // makes all instantiations of template pair<> friends

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

        // universal ctor
        template<typename U /*, typename std::enable_if_t<std::is_arithmetic_v<U>, bool> is_valid */>
        constexpr explicit pair(const ::pair<U>& other) noexcept :
            first(static_cast<T>(other.first)), second(static_cast<T>(other.second)) { }

        template<typename char_t> friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const pair& object) {
            // using function style casts
            ostream << char_t('{') << char_t(' ') << object.first << char_t(',') << char_t(' ') << object.second << char_t(' ')
                    << char_t('}');
            return ostream;
        }
};

int wmain() {
    constexpr ::pair<float> fpair { std::numbers::pi_v<float> };
    std::wcout << fpair << L'\n';

    constexpr auto spair { ::pair<short> {} };
    std::wcout << spair << L'\n';

    constexpr auto dpair {
        ::pair<double> { 12.086, 6543.0974 }
    };
    std::wcout << dpair << L'\n';

    constexpr auto z { dpair };
    ::pair<float>  q {};
    q = fpair;
    std::wcout << q << L'\n';

    constexpr ::pair<long> lpair { fpair }; // callsc the templated copy ctor
    std::wcout << lpair << L'\n';

    q = q;

    return EXIT_SUCCESS;
}
