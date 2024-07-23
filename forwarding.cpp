// g++ forwarding.cpp -Wall -Wextra -O3 -std=c++03 -municode

#include <cstdlib>
#include <string>
#include <vector>

// a C++03 style class
class nomoves {
        unsigned _counter;

    public:
        nomoves() throw() : _counter() { }

        nomoves(const unsigned& _init) throw() : _counter(_init) { }

        nomoves(const nomoves& other) throw() : _counter(other._counter) { ::_putws(L"copied!"); }

        nomoves& operator=(const nomoves& other) throw() {
            _counter = other._counter;
            return *this;
        }

#if __cplusplus >= 201103L // implement move semantics

        nomoves(nomoves&& other) throw() : _counter(other._counter) {
            ::_putws(L"moved!");
            other._counter = 0;
        }

        nomoves& operator=(nomoves&& other) throw() {
            _counter       = other._counter;
            other._counter = 0;
            return *this;
        }

#endif

        ~nomoves() throw() { _counter = 0; }

        nomoves operator+(const nomoves& other) const throw() { return nomoves(_counter + other._counter); }
};

int wmain() {
    // C++ did not have move semantics prior to C++11
    std::vector<nomoves> collection; // copy only in C++03

    nomoves       one(12);
    const nomoves two(45);

    collection.push_back(one);       // copy
    collection.push_back(two);       // copy
    collection.push_back(one + two); // create a temporary and copy it

    return EXIT_SUCCESS;
}
