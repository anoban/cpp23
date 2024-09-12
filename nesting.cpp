#include <cstdio>
#include <cstdlib>
#include <type_traits>

template<class T, class = std::enable_if<std::is_integral_v<T>, T>::type> class Integral final {
    private:
        T _value {};

    public:
        using value_type                               = T;

        constexpr Integral()                           = delete; // no default ctor
        constexpr Integral(const Integral&)            = delete; // no copy ctor
        constexpr Integral(Integral&&)                 = delete; // no move ctor
        constexpr Integral& operator=(const Integral&) = delete; // no copy assignment operator
        constexpr Integral& operator=(Integral&&)      = delete; // no move assignment operator

        constexpr explicit Integral(const T& init) noexcept : _value(init) { }
        constexpr ~Integral() = default;

        // conversion operator, to convert to arithmetic types
        template<class U, bool is_unsigned = std::is_unsigned<U>::value> requires std::is_arithmetic_v<U>
        constexpr operator U() const noexcept {
            return static_cast<U>(_value);
        }

        // templated copy ctor
        template<class U> requires std::is_arithmetic_v<U>
        constexpr Integral(const Integral<U>& other) noexcept : _value(static_cast<T>(other._value)) { }
};

// delete the conversion operator of Integral template when the required type is unsigned
template<class _Ty> template<class unsigned_type> Integral<_Ty> unsigned_type<unsigned_type, true>() noexcept = delete;

template<class T, class = std::enable_if<std::is_floating_point_v<T>, T>::type> class Floating final {
    private:
        T _value {};

    public:
        using value_type                               = T;

        constexpr Floating()                           = delete; // no default ctor
        constexpr Floating(const Floating&)            = delete; // no copy ctor
        constexpr Floating(Floating&&)                 = delete; // no move ctor
        constexpr Floating& operator=(const Floating&) = delete; // no copy assignment operator
        constexpr Floating& operator=(Floating&&)      = delete; // no move assignment operator

        constexpr explicit Floating(const T& init) noexcept : _value(init) { }
        constexpr ~Floating() = default;
};

auto wmain() -> int {
    constexpr ::Integral<short>          five46 { 546 };
    [[maybe_unused]] constexpr long long fivefour6 { five46 }; // implicit invocation of the conversion operator
    return EXIT_SUCCESS;
}
