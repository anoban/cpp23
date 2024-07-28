// g++ movsem_01.cpp -Wall -Wextra -Wpedantic -O3 -std=c++20 -o movsem_01.exe -municode

#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

using std::vector;

class f64
#if (__cplusplus >= 201103L)
    final // final keyword only available with C++11 and later
#endif
{

    private:
        long double _value;

    public:
        inline f64() throw() : _value(0.00L) { }

        inline explicit f64(const long double& _val) throw() : _value(_val) { }

        inline f64(const f64& other) throw() : _value(other._value) { ::_putws(L"copy construction"); }

        inline f64& operator=(const f64& other) throw() {
            _value = other._value;
            ::_putws(L"copy assignment");
            return *this;
        }

#if (__cplusplus >= 201103L) // move semantics

        inline f64(f64&& other) throw() : _value(other._value) {
            other._value = 0.00L;
            ::_putws(L"move construction");
        }

        inline f64& operator=(f64&& other) throw() {
            _value       = other._value;
            other._value = 0.00L;
            ::_putws(L"move assignment");
            return *this;
        }

#endif

        inline ~f64() throw() { _value = 0.00L; }

        f64 operator+(const f64& other) const throw() { return f64(_value + other._value); }

        long double& unwrap() throw() { return _value; }

        long double unwrap() const throw() { return _value; }
};

static inline vector<f64> test() throw() {
    vector<f64> container;
    container.reserve(4); // reserve storage for 4 f64 skeletons on heap
    // 1 extra ;p
    f64 pi(M_PI);
    container.push_back(pi);      // copy
    container.push_back(pi + pi); // create a temporary and copy in C++03 create a temporary and move in C++11 and later
    container.push_back(pi);      // copy

    return container;
}

int wmain() {
    const vector<f64> result = test();
    // no .cbegin() & .cend() members in C++03
    for (vector<f64>::const_iterator begin = result.begin(), end = result.end(); begin != end; ++begin)
#ifdef __GNUG__                                  // wprintf_s prints 0.0000 for long doubles with g++
        ::wprintf(L"%4.5Lf\n", begin->unwrap()); // NOLINT(cppcoreguidelines-pro-type-vararg)
#else
        ::wprintf_s(L"%4.5Lf\n", begin->unwrap()); // NOLINT(cppcoreguidelines-pro-type-vararg)
#endif

    return EXIT_SUCCESS;
}
