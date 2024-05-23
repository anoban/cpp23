#include <cstdlib>
#include <type_traits>

// handcrafted concept for signed integers
template<typename T> concept is_signed_integral = std::is_integral<T>::value && std::is_signed<T>::value;

// type constraint using requires expression
template<typename T> requires std::is_floating_point<T>::value // gratuitous trailing requires expression
static consteval T max(T _val0, T _val1) noexcept requires requires { _val0 > _val1; } {
    // using two successive requires keywords to mean that this overload requires the following requires expression
    return _val0 > _val1 ? _val0 : _val1;
}

// overload of max() for signed integers, using a requires clause
template<typename T> requires ::is_signed_integral<T> static consteval T max(const T& _sint0, const T& _sint1) noexcept {
    return _sint0 > _sint1 ? _sint0 : _sint1;
}

// a constrained overload of max for unsigned integers
template<typename T> requires requires {
    sizeof(T); // typename T should support these expressions
    static_cast<T>(1200);
}
static consteval T max(
    const T& _uint0, const T& _uint1, typename std::enable_if<std::is_unsigned<T>::value, T>::type = static_cast<T>(0)
) noexcept requires requires {
    _uint0 > _uint1;
    typename std::is_signed<T>::type;
    std::is_floating_point<T>::value; // see, these do not disqualify this overload's use with unsigned integers
                                      // requires expressions are evaluated only for syntactic correctness
} {                                   // only trailing requires expressions can use function arguments in their body
    return _uint0 > _uint1 ? _uint0 : _uint1;
}

template<typename T, typename = std::is_scalar<T>::type> class comparable {
    private:
        T value {};

    public:
        constexpr T get() const noexcept { return value; }

        constexpr comparable() noexcept = default;

        constexpr explicit comparable(const T& _init) noexcept : value { _init } { }

        constexpr ~comparable()                                 = default;

        // deliberately deleting copy and move ctors and copy and move assignment operators

        constexpr comparable(comparable& other)                 = delete;

        constexpr comparable(comparable&& other)                = delete;

        constexpr comparable operator=(const comparable& other) = delete;

        constexpr comparable operator=(comparable&& other)      = delete;

        template<typename U> constexpr bool operator<(const comparable<U>& other) const noexcept { return value < other.get(); }

        template<typename U> constexpr bool operator>(const comparable<U>& other) const noexcept { return value > other.get(); }

        // for equality and non-equality operators, types T and U must be same
        template<typename U> requires std::is_same<T, U>::value constexpr bool operator==(const comparable<U>& other) const noexcept {
            return value == other.value;
        }

        template<typename U> constexpr std::enable_if_t<std::is_same_v<T, U>, bool> operator!=(const comparable<U>& other) const noexcept {
            return value != other.value;
        }
};

// max overload for non-scalar types
template<typename T, typename U, typename = typename std::enable_if_t<!std::is_scalar_v<T> && !std::is_scalar_v<U>, bool>>
static constexpr auto max(const T& _nscalar0, const U& _nscalar1) noexcept requires requires {
    // type T must support all these equality comparisons
    _nscalar0 < _nscalar1;
    _nscalar0 > _nscalar1;
    // _nscalar0 == _nscalar1;
    // _nscalar0 != _nscalar1;
} {
    //  return _nscalar0 > _nscalar1 ? _nscalar0 : _nscalar1; ternary operator cannot deal with diffrent operand types
    // i.e when _nscalar0, _nscalar1 are of two different templated types
    if (_nscalar0 > _nscalar1) return _nscalar0;
    return _nscalar1;
}

// max overload for identical non-scalar templated types
template<typename T, typename = typename std::enable_if_t<!std::is_scalar_v<T>, bool>>
static constexpr auto max(const T& _nscalar0, const T& _nscalar1) noexcept requires requires {
    // type T must support all these equality comparisons
    _nscalar0 < _nscalar1;
    _nscalar0 > _nscalar1;
    _nscalar0 == _nscalar1;
    _nscalar0 != _nscalar1;
} {
    return _nscalar0 > _nscalar1 ? _nscalar0 : _nscalar1;
}

int wmain() {
    [[maybe_unused]] constexpr auto forty { ::max(12, 40) };
    [[maybe_unused]] constexpr auto ten { ::max(1.546562, 10.00000) };

    constexpr unsigned x { 11 };
    constexpr unsigned y { 33 };

    [[maybe_unused]] constexpr auto thirtythree { ::max(x, y) };

    constexpr auto five { ::comparable(5) };
    constexpr auto fifty { ::comparable(50) };
    constexpr auto seven { ::comparable(7.00F) };

    ::max(five, seven);
    ::max(five, fifty);

    return EXIT_SUCCESS;
}
