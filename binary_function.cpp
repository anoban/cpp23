//  g++ binary_function.cpp  -Wall -Wextra -Wpedantic -O3 -std=c++14

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include <sal.h>

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

        [[nodiscard]] constexpr result_type operator()(_In_ first_argument_type base, _In_ second_argument_type exponent) const throw() {
            T result { base };
            for (int i {}; i < exponent - 1; ++i) result *= base;
            return result;
        }
};

template<class T> struct square {
        using first_argument_type  = T;
        using second_argument_type = T;
        using result_type          = T;

        [[nodiscard]] constexpr result_type operator()(_In_ const first_argument_type value) const throw() { return value * value; }
};

static constexpr auto NELEMENTS { 12'000LLU };

int main() {
    std::random_device rdev {};	// for seeding the random number engine
    std::mt19937_64    rengine { rdev() };
    std::vector<float> randoms(NELEMENTS);
    std::vector<float> results(NELEMENTS);

    std::generate(randoms.begin(), randoms.end(), rengine);
    std::transform(randoms.cbegin(), randoms.cend(), results.begin(), square<float>{});

    // for (std::vector<float>::const_iterator it = randoms.cbegin(), end = randoms.cend(); it != end; ++it) ::wprintf_s(L"%.4f\t", *it);
    for(size_t i {}; i < randoms.size(); ++i) {
	::wprintf_s(L"%f^2 = %f\n", randoms.at(i), results.at(i));
}

    return EXIT_SUCCESS;
}
