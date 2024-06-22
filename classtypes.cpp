#include <cstdio>
#include <cstdlib>

template<typename T> struct type {
        using value_type = T;
        T _value {};

        constexpr explicit type(T init) noexcept : _value { init } { }

        constexpr T& get() noexcept { return _value; }

        constexpr const T& get() const noexcept { return _value; }

        constexpr ~type() = default;
};

auto wmain() -> int {
    constexpr auto object { type(9.000) };
    object.get();
    decltype(object)::value_type x {}; // x is just duble not const double!

    object._value = 10.00;
    
    return EXIT_SUCCESS;
}
