// C++03 did not have initializer lists!
// C++03 containers did not have cbegin(), cend() functions either
// g++ push_back.cpp  -Wall -Wextra -Wpedantic -O3 -std=c++03

#define _USE_MATH_DEFINES 1
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

struct printer {
        static unsigned count;

        template<class T> inline void operator()(const T& e) const {
            std::cout << e << ' ';
            count++;
        }
};

struct printerv2 {
        unsigned count;

        template<class T> inline void operator()(const T& e) {
            std::cout << e << ' ';
            count++;
        }
};

unsigned printer::count = 0;

int main() {
    std::vector<double> fvec(100, M_PI);
    for (std::vector<double>::const_iterator it = fvec.begin(); it != fvec.end(); it++) std::cout << *it << ' ';

    // std::vector<int>   invec { 10, 11, 12, 13, 14, 15 };   warning: extended initializer lists only available with '-std=c++11' or '-std=gnu++11'
    // emplace_back() to the rescue or push_back
    std::vector<int> invec(100);
    for (unsigned i = 0; i < invec.size(); ++i) invec.at(i) = rand() % 10; // emplace_back was not there in C++03
    for (std::vector<int>::const_iterator it = invec.begin(); it != invec.end(); it++) std::cout << *it << ' ';

    std::for_each(invec.begin(), invec.end(), printer());
    std::cout << printer::count << std::endl;

    const printerv2 ret = std::for_each(invec.begin(), invec.end(), printerv2());
    std::cout << ret.count << std::endl;
    return EXIT_SUCCESS;
}
