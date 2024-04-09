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
template<typename scalar_t>
requires std::integral<std::remove_reference_t<scalar_t>> || std::floating_point<std::remove_reference_t<scalar_t>>
// static constexpr scalar_t sq(scalar_t x) noexcept {  - this will expect references and rvalue references as return types
static constexpr std::remove_reference<scalar_t>::type sq(scalar_t x) noexcept {
    return x * x;
}

// template specialization overloads for long double
template<> static constexpr long double sq(long double x) noexcept { return x * x; }
// following specialization will lead to compile time errors if our templated return type is identical to the argument's type
// i.e scalar_t, so in order to make these work their return types should be explicitly changes to long double& and long double&&
// which is problematic because with long double& as retun type we'll be returning a reference to an object past its lifetime

// a better workaround is to remove the references from the return type with std::remove_reference_t<>
template<> static constexpr long double sq(long double& x) noexcept { return x * x; }
template<> static constexpr long double sq(long double&& x) noexcept { return x * x; }

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

    double         d = ::sq(3.1245F);       // calls sq<float>()

    double         e = ::sq(3.1245L);       // calls sq<long double>() specialization

    return 0;
}
