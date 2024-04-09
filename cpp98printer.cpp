// g++ cpp98printer.cpp  -Wall -Wextra -O3 -std=c++98 -Wpedantic

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

class printer {
        unsigned ncalls;

    public:
        printer() throw() : ncalls(0) { }

        // void operator()(const int x) noexcept { ::wprintf_s(L"%d\n", x); } - noexcept is a C++11 keyword
        void operator()(const int x) throw() {
            ::wprintf_s(L"%d ", x);
            ncalls++;
        }

        unsigned calls() const throw() { return ncalls; }
};

// we can also capture variables from the calling scope
struct greeter {
        std::wstring greeting; // variable to be captured by operator() from the calling scope
        unsigned     ncalls;

        explicit greeter(const std::wstring& str) : greeting(str), ncalls(0) { }

        void operator()(int x) throw() {
            ::wprintf_s(L"%s, %u\n", greeting.c_str(), x);
            ++ncalls;
        }

        unsigned count() const throw() { return ncalls; }
};

int main() {
    // std::vector<int> v { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; - uniform initializer list is a C++11 feature
    std::vector<int> v(10);
    int              x = 1;

    // C++98 doesn't support range based loops either :(
    for (std::vector<int>::iterator it = v.begin(), end = v.end(); it != end; ++it) {
        *it = x;
        ++x;
    }

    // std::for_each(v.cbegin(), v.cend(), printer {}); .cbegin and .cend methods are unavailable in C++98
    // and braced initialization is unavailable for ctors
    std::for_each(v.begin(), v.end(), printer());
    const printer out = std::for_each(v.begin(), v.end(), printer());

    // only std::foreach returns the mutated functor

    ::wprintf_s(L"printer has been called %u times!\n", out.calls());
    printer().operator()(2387); // temporary

    const greeter greet = std::for_each(v.begin(), v.end(), greeter(L"Hi there! "));
    ::wprintf_s(L"greeter has been called %u times!\n", greet.count());

    return 0;
}
