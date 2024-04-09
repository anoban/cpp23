#include <cstdlib>

static consteval unsigned five() noexcept { return 5; }

// when creating an object of type cmptime, x must be a compiletime constant
template<size_t x> struct cmptime { };

int main() {
    const size_t     x = rand();
    constexpr size_t xx { 45 };

    cmptime<xx>      works {}; // because xx is a compile time constant
    cmptime<x> wont {}; // because x is NOT A COMPILE TIME CONSTANT EXPRESSION! IT IS A CONSTANT STILL BUT.... NOT KNOWN AT COMPILETIME
    // Error: non - type template argument is not a constant expression cmptime<five()> will {};

    cmptime<five() + 8657> will_too {};   // five() is a consteval function, so okay
    cmptime<rand()>        mhmmm_nope {}; // Error: non-type template argument is not a constant expression

    const long             xxx { 756 };
    static const unsigned  _x_ { 65 };
    cmptime<xxx>           okay {}; // template arguments are compile time constant expressions, just plain integer literals
    cmptime<_x_>           okay_too {};

    return EXIT_SUCCESS;
}
