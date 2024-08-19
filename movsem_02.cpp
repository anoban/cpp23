// g++ movsem_02.cpp -Wall -Wextra -Wpedantic -O3 -std=c++xx -o movsem_02.exe -I ./
// <cstdint> requires C++11 and later with g++ hence won't compile with -std=c++03

#include <sstring>

static inline ::sstring skyfall() __noexcept__ {
    // NOLINTNEXTLINE(modernize-return-braced-init-list)
    return ::sstring("Skyfall is where we start, a thousand miles and poles apart, where worlds collide and days are dark!");
}

static const size_t MiB = 1024 * 1024;

int main() {
    const ::sstring empty; // default construction
    ::sstring       rust_style = ::sstring::with_capacity(7 * MiB);

    const ::sstring jbond(" I've drowned and dreamt this moment.... so overdue I owe them................");
    std::cout << jbond << '\n';

    sstring adele;       // default construction
    adele = ::skyfall(); // copy assignment in C++03, move assignment in C++11 and later

    std::cout << adele << '\n';

    const sstring aaaaa('A', 50);
    std::cout << aaaaa << '\n';

    const ::sstring concatenated = skyfall() + jbond;
    std::cout << concatenated << '\n';

    return EXIT_SUCCESS;
}
