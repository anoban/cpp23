// clang .\undecorate.cpp -O0 -c -std=c++20 -Wall -Wextra -pedantic

namespace mangle {
    struct mangled {
            double __stdcall operator()() const throw();
    };
} // namespace mangle

double __stdcall mangle::mangled::operator()() const throw() { return 123456789.0; }
