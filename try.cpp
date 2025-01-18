#include <iostream>
#include <stdexcept>

[[nodiscard, maybe_unused]] static constexpr double factorial(unsigned _integer) noexcept(false) {
    if (_integer > 30) throw std::runtime_error("cannot work with values bigger than 30!");
    double result { 1.00000 };
    while (_integer) result *= _integer--;
    return result;
}

static_assert(::factorial(10) == 3628800);

// static_assert(::factorial(50)); // throws

// a function could just be a try catch block
[[nodiscard, maybe_unused]] static constexpr double caller(const unsigned& _value) noexcept(noexcept(::factorial(_value))) try {
    return ::factorial(_value);
} catch (const std::runtime_error& rt_err) {
    std::cout << rt_err.what() << std::endl; // NOLINT
    return 0.0000;
}

// alternative implementation
[[nodiscard, maybe_unused]] static constexpr double invoke(const unsigned& _value) noexcept(noexcept(::factorial(_value))) {
    try {
        return ::factorial(_value);
    } catch (const std::runtime_error& rt_err) {
        std::cout << rt_err.what() << std::endl; // NOLINT
        return 0.0000;
    } catch (const std::exception& excpt) {
        std::cout << excpt.what() << std::endl; // NOLINT
        return -1.0000;
    } catch (...) { return -100.000; }
}

auto main() -> int {
    for (long i = 23; i < 450; ++i) {
        std::cout << ::caller(i) << '\n';
        std::cout << ::invoke(i) << '\n';
    }

    return EXIT_SUCCESS;
}

template<typename _Ty> requires std::is_arithmetic_v<_Ty>
[[nodiscard]] static constexpr long double power(const _Ty& _mantissa, unsigned long _exponent) noexcept(false) {
    long double init { 1.0000 };
    while (_exponent) {
        init *= _mantissa;
        _exponent--;
    }
    return init;
}

static_assert(::power(4.6564, 73)); // must be 5.8569360348248E+48
