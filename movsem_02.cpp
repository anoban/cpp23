// g++ movsem_02.cpp -Wall -Wextra -Wpedantic -O3 -std=c++xx -o movsem_02.exe -I ./
// <cstdint> requires C++11 and later with g++ hence won't compile with -std=c++03

#include <string.hpp>

using value_semantics::string;

static inline ::string skyfall() throw() {
    return ::string("Skyfall is where we start, a thousand miles and poles apart, where worlds collide and days are dark");
}

static const size_t MiB = 1024 * 1024;

int main() {
    const ::string empty; // default construction
    ::string       onemib = ::string::with_capacity(7 * MiB);

    const ::string jbond("I've drowned and dreamt this moment.... so overdue I owe them................");
    ::puts(jbond.c_str());

    string adele;        // default construction
    adele = ::skyfall(); // copy assignment in C++03, move assignment in C++11 and later

    ::puts(adele.c_str());

    const string aaaaa('A', 50);
    ::puts(aaaaa.c_str());

    return EXIT_SUCCESS;
}
