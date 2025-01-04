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

[[nodiscard, maybe_unused]] static constexpr double caller(const unsigned& _value) noexcept(noexcept(::factorial(_value))) try {
    return ::factorial(_value);
} catch (const std::runtime_error& rt_err) {
    std::cout << rt_err.what() << std::endl;
    return 0.0000;
}

auto main() -> int {
    for (long i = 23; i < 100; ++i) std::cout << ::caller(i) << '\n';
    return EXIT_SUCCESS;
}
