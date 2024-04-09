// clang .\funtionhelpers.cpp -Wall -O3 -std=c++20 -Wextra -pedantic

#include <functional>
#include <iostream>
#include <numbers>

// function with two arguments
template<typename T> static constexpr T sum(T x, T y) noexcept { return x + y; }

// a function that takes three arguments
static double                           sumthree(double a, double b, double c) noexcept {
    ::wprintf_s(L"a = %lf, b = %lf, c = %lf\n", a, b, c);
    return a + b + c;
}

int main() {
    constexpr float x { 6.7834576 }, y { 3.22345872334 };
    // <functional> brings an array of function helpers (functors)
    // i.e std::plus<T>{ } constructs a functor
    // and std::plus<T>{ }() uses that functor's operator()
    constexpr auto  sum { std::plus<float>()(x, y) };
    std::wcout << x << L' ' << y << L" = " << sum << L'\n';

    constexpr auto twopi { std::multiplies<long double> {}(std::numbers::pi_v<long double>, 2.00L) };
    std::wcout << L" 2 pi = " << twopi << L'\n';

    constexpr auto is_two_lessthan_one { std::less<int>()(2, 1) };
    std::wcout << L"is 2 < 1 ? " << std::boolalpha << is_two_lessthan_one << L'\n';

    constexpr auto add_ten {
        std::bind(::sum<float>, std::placeholders::_1, 10) // add_ten() will use 10 as implicit second argument
    }; // std::bind1st, std::bind2nd functions were deprecated in C++11 removed in C++17
    std::wcout << L"24 + 10 = " << add_ten(24) << L'\n';

    // a variant of sumthree function bound to 10 and 20 as the second and third arguments
    // argument passed to add10and20 will be received as the first argument
    constexpr auto add10and20 { std::bind(sumthree, std::placeholders::_1, 10, 20) };
    add10and20(17);

    constexpr auto add100 = std::bind_front(sumthree, 100.00);
    add100(78.05, 54.68754);

    constexpr auto diff = std::bind_front(std::minus<float> {}, 23.0000);
    // std::minus{}(x, y) returns x - y
    std::wcout << L"23.000 - 45984.87752476 is " << diff(45984.87752476) << L'\n';

    // std::less evaluates whether thr first argument is less than the second argument
    constexpr auto is_lessthan100 = std::bind(std::less<double> {}, std::placeholders::_1, 100.000);
    std::wcout << L"Is 67345.764131 less than 100.000? " << std::boolalpha << is_lessthan100(67345.764131) << L'\n';

    return EXIT_SUCCESS;
}
