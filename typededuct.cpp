namespace _pass_by_value {

    template<typename T> constexpr double factorial(T n) noexcept { return n ? n * factorial(n - 1) : 1; }

    static_assert(factorial(0) == 1);
    static_assert(factorial(9) == 362880);
    static_assert(factorial(13) == 6227020800);

} // namespace _pass_by_value

namespace _pass_by_pointer {
    template<typename T> constexpr double factorial(T* n) noexcept { return (*n) ? (*n) * _pass_by_value::factorial((*n) - 1) : 1; }

    static const unsigned                 a { 0 }, b { 9 }, c { 13 };

    static_assert(factorial(&a) == 1);
    static_assert(factorial(&b) == 362880);
    static_assert(factorial(&c) == 6227020800);
} // namespace _pass_by_pointer
