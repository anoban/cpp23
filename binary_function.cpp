//  g++ binary_function.cpp  -Wall -Wextra -Wpedantic -O3 -std=c++14

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// base class
template<class _Arg1, class _Arg2, class _Result> struct binary_function {
        using first_argument_type  = _Arg1;
        using second_argument_type = _Arg2;
        using result_type          = _Result;
};

template<class T> struct power /* : public binary_function<T, T, T> */ {
        // or redefine local type aliases
        using first_argument_type  = T;
        using second_argument_type = T;
        using result_type          = T;

        [[nodiscard]] constexpr result_type operator()(_In_ const T& base, _In_ const T& exponent) const throw() {
            T result { base };
            for (T i { 1 }; i < exponent; ++i) result *= base;
            return result;
        }
};

template<class T> struct square {
        using first_argument_type  = T;
        using second_argument_type = T;
        using result_type          = T;

        [[nodiscard]] constexpr T operator()(_In_ const T value) const throw() { return value * value; }
};

static constexpr auto NELEMENTS { 120'000 };

template<typename _arg0_type, typename _arg1_type, typename _arg2_type, typename _res_type = long long> struct lambda {
        constexpr _res_type operator()(const _arg0_type& _arg0, const _arg1_type& _arg1, const _arg2_type& _arg2) noexcept {
            return _arg0 + _arg1 + _arg2;
        }
};

int main() {
    std::random_device rdev {}; // for seeding the random number engine
    std::mt19937_64    rengine { rdev() };

    std::vector<int>              randoms(NELEMENTS);
    std::vector<int>              results(NELEMENTS);
    decltype(randoms)::value_type x {};
    size_t                        sneaky {};

    std::generate(randoms.begin(), randoms.end(), [&rengine, &sneaky]() noexcept -> decltype(rengine()) {
        const auto rnd  = rengine() % 4;
        sneaky         += rnd * rnd;
        return rnd;
    });
    std::generate(results.begin(), results.end(), [&rengine]() noexcept { return rengine() % 4; });

    const auto sum { std::reduce(randoms.cbegin(), randoms.cend(), 0LLU, std::plus<> {}) };
    std::transform(randoms.cbegin(), randoms.cend(), randoms.begin(), square<int> {});
    const auto sumsq { std::reduce(randoms.cbegin(), randoms.cend(), 0LLU, std::plus<> {}) };

    std::wcout << L"sum = " << sum << L" and the sum of squares is " << sumsq << L" sneaky sum is " << sneaky << std::endl;

    const auto rsum { std::accumulate(results.cbegin(), results.cend(), 0LLU) };
    const auto quadruple { std::bind(::power<size_t> {}, std::placeholders::_1, 4LLU) };
    std::transform(results.cbegin(), results.cend(), results.begin(), quadruple);
    const auto qrsum { std::accumulate(results.cbegin(), results.cend(), 0LLU) };

    std::wcout << L"rsum = " << rsum << L" and the sum of quadruples is " << qrsum << std::endl;

    // for (std::vector<float>::const_iterator it = randoms.cbegin(), end = randoms.cend(); it != end; ++it) ::wprintf_s(L"%.4f\t", *it);
    // for (size_t i {}; i < randoms.size(); ++i) ::wprintf_s(L"%f^2 = %f\n", randoms.at(i), results.at(i));

    constexpr auto increment = [](auto& e) consteval noexcept -> void { e++; };
    constexpr auto incr      = [](auto& e) consteval noexcept -> void { e++; };

    float f {};

    static_assert(!std::is_same_v<decltype(increment), decltype(incr)>);
    static_assert(std::is_same_v<void, decltype(incr(f))>);

    return EXIT_SUCCESS;
}
