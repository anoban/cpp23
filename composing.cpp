// g++ composing.cpp -Wall -Wextra -O3 -std=c++11 -Wpedantic

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>

static const size_t NELEMENTS { 100 };

// a std::binary_function type closure object
template<typename T> struct less /* : public std::binary_function<T, T, bool> or just : public std::binary_function<T> */ {
        typedef T    first_argument_type;
        typedef T    second_argument_type;
        typedef bool result_type;

        constexpr result_type operator()(first_argument_type x, second_argument_type y) const throw() { return x < y; }
};

int main() {
    auto predicate_lt    = std::bind(less<int>(), std::placeholders::_1, 250);        // less than 250
    auto predicate_gt    = std::bind(std::greater<int>(), std::placeholders::_1, 17); // greater than 17

    auto predicate_lt_v2 = std::bind2nd(less<int>(), 250);        // less than 250
    auto predicate_gt_v2 = std::bind1st(std::greater<int>(), 17); // greater than 17

    std::array<int, NELEMENTS> arr {};
    int                        x { 12 }; // unnecessary

    for (std::array<int, NELEMENTS>::iterator it = arr.begin(); it != arr.end(); ++it, ++x) *it = x; // unnecessary

    // less than 250 and greater than 17
    auto composed_predicate    = std::bind(std::logical_and<bool>(), predicate_lt, predicate_gt);
    auto composed_predicate_v2 = std::bind(std::logical_and<bool>(), predicate_lt_v2, predicate_gt_v2);

    const auto count           = std::count_if(arr.cbegin(), arr.cend(), composed_predicate);
    // const auto count_v2              = std::count_if(arr.cbegin(), arr.cend(), composed_predicate_v2);

    ::wprintf_s(L"What about 9? %d\n", composed_predicate_v2(9));

    ::wprintf_s(L"Between 12 and %d, %zu numbers were greater than 17 and less than 250\n", x, count);
    return EXIT_SUCCESS;
}
