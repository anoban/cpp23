#include <concepts>

// demonstrating ambiguity in C++
// double x = sq(3.1245);
// what could the above statement possibly mean?

// sq could be a macro
#define sq(expression) ((expression) * (expression))
#undef sq // we do not want the macro troubling us downstream.

// sq could be a regular function
static double sq(double x) noexcept { return x * x; }

// a constrained templated function
template<typename scalar_t> requires std::integral<scalar_t> || std::floating_point<scalar_t>
static constexpr scalar_t sq(scalar_t x) noexcept {
    return x * x;
}

// a class type
struct sq {
        double value; // stores the squared value

        sq() = delete;

        // ctor
        explicit constexpr sq(double x) noexcept : value { x * x } { }

        // function style cast from sq class to double
        // auto object = sq(12.675376);
        //
        explicit constexpr operator double() const noexcept { return value; }
};

// a functor
struct sqGenerator {
        constexpr double operator()(double x) const noexcept { return x * x; }
};

int main() {
    constexpr auto sq { sqGenerator {} };
    double         a = sq(3.1245);          // calls the operator() of object sq

    double         b = ::sq(3.1245);        // freestanding function

    double         c = ::sq<float>(3.1245); // calls the templated function

    return 0;
}
