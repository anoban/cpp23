#include <string>

// templates can be seen as a metafunction that maps a string of parameters to a function or a class

template<typename T> [[nodiscard]] static T add(const T& a, const T& b) { return a + b; }

// in addition to std::enable_if<> and concepts, there is a crude way to impose type restrictions on templated entities
// declare the templated entity without providing a definition and then implement specializations to support only the types you want

template<typename T> constexpr T            sum(T x, T y) noexcept;                                              // declaration
template<> constexpr int                    sum(int x, int y) noexcept { return x + y; }                         // specialization for int
template<> constexpr short                  sum(short x, short y) noexcept { return static_cast<short>(x + y); } // specialization for short
// sum will not work with any types besides the ones the specializations are provided for.

template<typename T> constexpr T            power(T value, unsigned char exp) noexcept {
    T result { value };
    for (unsigned i = 0; i < exp; ++i) result *= result;
    return result;
}

// templates accept two types of arguments - types and concrete values
// all of these types and value need to be compile time constants

int main() {
    const std::wstring name { L"Julia" };
    const std::wstring father { L"Edwards" };

    // here the signature T add(const T&, const T&) gets translated into std::wstring add(const std::wstring&, const std::wstring&)
    // practically all T placeholders get substituted with a concrete type
    const auto         full { add(name, father) };

    constexpr auto     s  = sum(54, 87345);     // okay int
    constexpr auto     ss = sum(656I16, 54I16); // okay short

    // link time errors - 6 unresolved externals
    sum(9.546325F, 98.785675F); // float
    sum(9.546325L, 98.785675L); // long double
    sum(9.546325, 98.785675);   // double
    sum(9LLU, 9875LLU);         // unsigned long long
    sum(954U, 9878U);           // unsigned
    sum(9546LL, 98LL);          // long long

    // when templated functions are called without explicit template arguments, the compiler will try to deduce them
    const auto two_exp8   = ::power(2, 8); // compiler deduced the return type from the type of the first argument
    const auto dbl_exp8   = ::power(2.000, 8);
    const auto float_exp8 = ::power(2.000F, 8);
    const auto ldbl_exp8  = ::power(2.000L, 8);
    const auto byte       = ::power<unsigned char>(2, 8); // explicit template argument
    // explicit template arguments can be used to overcome overload resolution conflicts
    // or to coerce type promotions or truncations

    return EXIT_SUCCESS;
}

template<typename T /* T here is a template parameter */> requires std::convertible_to<T, double>
static constexpr double cube(T x) noexcept {
    return static_cast<double>(x * x * x);
}

// arguments passed to a templated entity at call site is called template arguments
constexpr double coobe(long long a) noexcept { return cube<long long /* template argument */>(a); }

// when the compiler has successfully deduced the template arguments or when the template arguments have been explicitly provided,
// the template will be instantiated.
