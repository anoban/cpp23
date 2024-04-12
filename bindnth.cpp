#include <functional>
#include <iomanip>
#include <iostream>

// g++ bindnth.cpp -Wall -Wextra -Wpedantic -std=c++11 -O3
// g++ bindnth.cpp -Wall -Wextra -Wpedantic -std=c++20 -O3

// takes 7 arguments of different types
static double sum(short s, char c, unsigned u, long long ll, float f, double d, long double ld) noexcept {
    return static_cast<double>(s + c + u + ll + f + d + ld);
}

static double bsum(short s, long l) noexcept { return static_cast<double>(s + l); }

// std::bind_front, std::bind_back, std::bind1st & std::bind2nd do not accept placeholders

const auto    ___sum = std::bind(
    sum, static_cast<short>(20), 'A', 20U, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4
);

#if (__cplusplus <= 201103L)
// AdaptableBinaryFunction :=
template<typename scalar_t>
// if the parent class type template receives the arguments, upower will inherit them
// if not i.e if upower is defined as struct upower : public std::binary_function {}, then the types need to be defined inside the inheriting class type
struct upower : public std::binary_function<scalar_t /* typename _Arg1 */, unsigned /* typename _Arg2 */, scalar_t /* typename _Result */> {
        // AdaptableBinaryFunction is basically a functor that inherits from std::binary_function

        // these are standard typedefs expected of an AdaptableBinaryFunction
        // typedef unsigned second_argument_type;
        // typedef scalar_t first_argument_type;
        // typedef scalar_t result_type;

        scalar_t operator()(scalar_t base, unsigned exp) const noexcept { // only accepts unsigned integer exponents!
            if (!exp || exp == 1) return base;
            scalar_t result = 1;
            for (unsigned i = 0; i < exp; ++i) result *= base;
            return result;
        }
};
#endif

int main() {
    std::wcout << std::fixed << std::setprecision(5);

    const short     SHORT { 100 };
    const char      CHAR { 'e' }; // 100
    const unsigned  UNSIGNED { 100 };
    const long long LONGLONG { 100 };
    const float     FLOAT { 100.000 };
    const float     DOUBLE { 100.000 };
    const float     LONGDOUBLE { 100.000 };

#if defined(__cplusplus) && (__cplusplus >= 202002L) // C++20 and later
    const auto __sum = std::bind_front(sum, static_cast<short>(20), 'A');
    std::wcout << __sum(UNSIGNED, LONGLONG, FLOAT, DOUBLE, LONGDOUBLE) << L'\n';
#endif

    // lol std::bind managed to survive through all these standard revisions
    std::wcout << ___sum(LONGLONG, FLOAT, DOUBLE, LONGDOUBLE) << L'\n';

#if defined(__cplusplus) && (__cplusplus <= 201103L) // C++11 and before

    // std::bind1st and std::bind2nd can only be used with AdaptableBinaryFunctions
    // these are not regular functions and plain binary functions need to be converted to an AdaptableBinaryFunction using std::ptr_fun

    // std::bind1st binds the default argument to the first argument
    const auto bsum_ = std::bind1st(std::ptr_fun(&bsum), static_cast<short>(40));

    // std::bind2nd binds the default argument to the second argument
    const auto _bsum = std::bind2nd(std::ptr_fun(&bsum), static_cast<long>(40));

    std::wcout << _bsum(SHORT) << L'\n';
    std::wcout << bsum_(100L) << L'\n';

    const auto square = std::bind2nd(upower<float> {}, 2);
    std::wcout << L"pi^2 = " << square(3.14159265358979F) << L'\n';

    const auto piToPow = std::bind1st(upower<double> {}, 3.14159265358979);
    for (unsigned i = 0; i < 11; ++i) std::wcout << L"pi^" << i << L" = " << piToPow(i) << L'\n';

#endif

    return EXIT_SUCCESS;
}

// starting to feel really comfortable with C++ :)
