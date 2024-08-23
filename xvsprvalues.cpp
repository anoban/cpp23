// references and temporaries
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>

using std::string;

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

int main() {
    const auto pi                     = ::function('A');

    // reference to temporaries
    // [[maybe_unused]] float&        x  = M_PI; // prvalue cannot bind to a lvalue reference type
    [[maybe_unused]] const float&  y  = M_PI; // prvalue can bind to a const lvalue reference
    [[maybe_unused]] float&&       z  = M_PI; // prvalue can bind to a rvalue reference
    [[maybe_unused]] const float&& $  = M_PI; // prvalue can also bind to a const rvalue reference

    // type punning is not allowed in reference bnding unless the reference is a const reference
    // i.e when the reference will only be used for read opeations

    int          age                  = 27; // 4 bytes
    // float&       refage              = age; // error
    const float& crefage              = age; // okay because the compiler creates a temporary of float type and binds that to the reference
    // hence mutating the reference will not modify the refererred to object
    // thus, to disguise the use of using a copy behind the scenes, cross type reference binding is allowed only when the reference is a const
    // reference, so users will not naturally attempt to modify them

    // so crefage is a const reference to a float with value 27.0

    float* crefptr                    = const_cast<float*>(&crefage);
    *crefptr                         *= 2.00; // mutating an allegedely const float

    ::printf_s("age = %5d, mutated copy = %.10f\n", age, crefage);

    float* ptr = reinterpret_cast<float*>(&age); // won't work without the cast
    // const float* cptr                = &age;                           // still an error

    const string name { "Anoban" };

    return EXIT_SUCCESS;
}
