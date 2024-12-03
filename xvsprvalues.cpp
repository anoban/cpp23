// references and temporaries
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>

using std::string;
using std::wstring;

template<bool predicate, class T> struct enable_if { };

template<class T> struct enable_if<true, T> {
        static const bool value = true;
        typedef T         type;
};

template<class T>
static __declspec(noinline) typename ::enable_if<std::is_arithmetic<T>::value, T>::type function(_In_ [[maybe_unused]] const T& x
) noexcept {
    return M_PI;
}

class real_wrapper;

class integer_wrapper final {
        long long _value;

    public:
        constexpr explicit integer_wrapper(const int& value) noexcept : _value { value } { }

        constexpr long long value() const noexcept { return _value; }

        integer_wrapper(const real_wrapper& other);
};

class real_wrapper final {
        double _value;

    public:
        constexpr explicit real_wrapper(const double& value) noexcept : _value { value } { }

        constexpr double value() const noexcept { return _value; }

        real_wrapper(const integer_wrapper& other) noexcept : _value { static_cast<double>(other.value()) } { }
};

integer_wrapper::integer_wrapper(const real_wrapper& other) noexcept : _value { static_cast<long long>(other.value()) } { }

int main() {
    const auto                     pi  = ::function('A');

    // reference to temporaries
    // [[maybe_unused]] float&        x  = M_PI; // prvalue cannot bind to a lvalue reference type
    [[maybe_unused]] const float&  y   = M_PI; // prvalue can bind to a const lvalue reference
    [[maybe_unused]] float&&       z   = M_PI; // prvalue can bind to a rvalue reference
    [[maybe_unused]] const float&& $   = M_PI; // prvalue can also bind to a const rvalue reference

    // type punning is not allowed in reference bnding unless the reference is a const reference
    // i.e when the reference will only be used for read opeations

    int                            age = 27; // 4 bytes
    // float&       refage              = age; // error
    const float& crefage               = age; // okay because the compiler creates a temporary of float type and binds that to the reference
    // hence mutating the reference will not modify the refererred to object
    // thus, to disguise the use of using a copy behind the scenes, cross type reference binding is allowed only when the reference is a const
    // reference, so users will not naturally attempt to modify them

    // so crefage is a const reference to a float with value 27.0

    float*       crefptr               = const_cast<float*>(&crefage);
    *crefptr *= 2.00; // mutating an allegedely const float

    ::printf_s("age = %010X, crefage = %010X\n", &age, &crefage);
    ::printf_s("age = %5d, mutated copy = %.10f\n", age, crefage);

    float*              ptr = reinterpret_cast<float*>(&age); // won't work without the cast
        // const float* cptr                = &age;                           // still an error

    integer_wrapper     ten { 10 };
    const real_wrapper& not_ten { ten }; // not a type punned reference but a reference to a completely different object
    ::printf_s("ten = %010X, not_ten = %010X\n", &ten, &not_ten);

    return EXIT_SUCCESS;
}
