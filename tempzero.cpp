#include <cstddef>
#include <cstdlib>
#include <type_traits>

template<int x> struct with_default {
        static consteval int func(const int _default = 0) noexcept { return x / _default; }

        constexpr int        operator()(const int v = 10) const noexcept { return x / v; }
};

template<typename T, double (*sum)(const T x, const T y) noexcept> struct summer {
        constexpr double operator()(const T a, const T b, typename std::enable_if<std::is_scalar<T>::value, T>::type = static_cast<T>(0))
            const noexcept {
            return (*sum)(a, b);
        }
};

template<size_t x> static constexpr size_t   f() noexcept { return x / x; }

template<typename T> static constexpr double s(const T arg_0, const T arg_1) noexcept { return arg_0 + arg_1; }

int                                          main() {
    //
    constexpr auto okay { ::with_default<12>::func(12) };       // 12 / 12
    constexpr auto nope { ::with_default<12>::func() };         // 12 / 0
    constexpr auto mhmm { ::with_default<0> {}.operator()(0) }; // 0 / 0
    constexpr auto okayy { ::with_default<0> {}.operator()() }; // 0 / 10

    auto           what { ::f<0>() };         // no compile time errors!
    constexpr auto thats_better { ::f<0>() }; // compile time errors because of constexpr

    auto           error { ::f<0LLU / 0>() };
    auto           not_error { ::f<0LLU / 10>() };

    constexpr auto x { ::summer<float, s> {}(2.5F, 2.000F) }

    return EXIT_SUCCESS;
}
