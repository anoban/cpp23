// g++ forwarding.cpp -Wall -Wextra -O3 -std=c++03 -municode
// g++ forwarding.cpp -Wall -Wextra -O3 -std=c++11 -municode

#include <cstdlib>
#include <string>
#include <vector>

// a C++03 style class
class nomoves {
        unsigned _counter;

    public:
        nomoves(); // C++03 compatible way of deleting a ctor overload

        nomoves(const unsigned& _init) throw() : _counter(_init) { ::_putws(L"created!"); }

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

#endif // __cplusplus >= 201103L

        ~nomoves() throw() { _counter = 0; }

        nomoves operator+(const nomoves& other) const throw() { return nomoves(_counter + other._counter); }
};

int wmain() {
    // C++ did not have move semantics prior to C++11
    std::vector<nomoves> collection; // copy only in C++03
    collection.reserve(20);          // to avoid reallocations

    nomoves       one(12);
    const nomoves two(45);

    collection.push_back(one);       // copy in all standards
    collection.push_back(two);       // copy in all standards
    collection.push_back(one + two); // create a temporary and copy it in C++03 and before create a temporary and move it in C++11 & later

    ::_putws(L"yeehaww!");

    // when a reallocation happens, std::vector re constructs all of its elements FUCK??
    collection.reserve(100); // expect three new copies in C++03 and three moves in C++11 and later

    constexpr auto const _ptr = "What's your problem?";
    constexpr auto       y    = 8 [_ptr];
    constexpr unsigned   array[10] { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2 };

    constexpr auto what = 7 [array];

    return EXIT_SUCCESS;
}
