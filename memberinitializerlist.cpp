#include <cmath>
#include <iostream>
#include <numbers>

constexpr float pi(long double& inout) noexcept { return static_cast<float>(inout = std::numbers::pi_v<long double>); }

constexpr float Z(long double& inout) noexcept { return static_cast<float>(inout = 'Z'); }

struct record   final {
        unsigned      a;
        float         b;
        long double   c;
        unsigned char letter;

        constexpr record() noexcept :
            a { 10U }, b { ::pi(c) /* this will initialize the member variable c too */ }, c { /* overwrite c? */ }, letter { 'X' } { }

        inline record(const unsigned& _a, const unsigned char& _letter) noexcept :
            a { _a }, b { ::pi(c) }, /* c is not overwritten by this ctor */ letter { _letter } { }

        inline explicit record([[maybe_unused]] const char&) noexcept :
            a { 10U }, b { ::Z(c) }, /* c is not overwritten here */ letter { static_cast<unsigned char>(c) } { }

        friend inline std::ostream& operator<<(std::ostream& ostr, const record& rec) noexcept(noexcept(ostr << ' ')) {
            ostr << rec.a << ' ' << rec.b << ' ' << rec.c << ' ' << rec.letter << '\n';
            return ostr;
        }
};

int main() {
    // constexpr record one { 67U, 'A' }; // clang won't compile this, error: constexpr variable 'one' must be initialized by a constant expression
    const record one { 67U, 'A' };
    std::cout << one;

    const record overwritten {};
    std::cout << overwritten;

    const record complicated { 'Z' };
    std::cout << complicated;

    return EXIT_SUCCESS;
}
