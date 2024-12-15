#include <cstdlib>

static consteval unsigned five() noexcept { return 5; }

// when creating an object of type mystruct, x must be a compiletime constant
template<size_t x> struct mystruct { };

int main() {
    const size_t     x = rand();
    constexpr size_t xx { 45 };

    mystruct<xx> works {}; // because xx is a compile time constant
    mystruct<x>  wont {};  // because x is NOT A COMPILE TIME CONSTANT EXPRESSION! IT IS A CONSTANT STILL BUT.... NOT KNOWN AT COMPILETIME
    // Error: non - type template argument is not a constant expression mystruct<five()> will {};

    mystruct<five() + 8657> will_too {};   // five() is a consteval function, so okay
    mystruct<rand()>        mhmmm_nope {}; // Error: non-type template argument is not a constant expression

    const long            xxx { 756 };
    static const unsigned _x_ { 65 };
    mystruct<xxx>         okay {}; // template arguments are compile time constant expressions, just plain integer literals
    mystruct<_x_>         okay_too {};

    // for value type template arguments, division by 0 causes a compilation error
    mystruct<(54 + 76) * (4 / 0)> divbyzero {}; // Error: division by zero

    // non-compile time evaluated function calls are forbidden
    // saw that already
    mystruct<rand()> error {}; // Error: non-constexpr function 'rand' cannot be used in a constant expression

    // expressions that produce non-integer or non-pointer types are non-portable
    mystruct<static_cast<unsigned>(x * 8.0564564)> meh {};

    return EXIT_SUCCESS;
}
