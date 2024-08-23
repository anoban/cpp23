// g++ movsem_02.cpp -Wall -Wextra -Wpedantic -O3 -std=c++xx -o movsem_02.exe -I ./
// <cstdint> requires C++11 and later with g++ hence won't compile with -std=c++03

#include <sstring>

static inline ::sstring skyfall() noexcept {
    // NOLINTNEXTLINE(modernize-return-braced-init-list)
    return ::sstring("Skyfall is where we start, a thousand miles and poles apart, where worlds collide and days are dark!");
}

static __declspec(noinline) ::sstring no_time_to_die() noexcept { return "That I've fallen for a lie.... you were never on my side...."; }

static const size_t MiB = 1024 * 1024;

int main() {               // NOLINT(bugprone-exception-escape)
    const ::sstring empty; // default construction
    ::sstring       crab_style = ::sstring::with_capacity(7 * MiB);

    const ::sstring jbond("I've drowned and dreamt this moment.... so overdue I owe them................");
    std::cout << jbond << '\n';

    sstring adele;       // default construction
    adele = ::skyfall(); // copy assignment in C++03, move assignment in C++11 and later

    std::cout << adele << '\n';

    const sstring aaaaa('A', 50);
    std::cout << aaaaa << '\n';

    const ::sstring concatenated = skyfall() + " " + jbond; // boy look at that :)
    std::cout << concatenated << '\n';

    const ::sstring no_time_to_die = ::no_time_to_die(); // move construction in C++11 and later, copy construction in in C++03 and before
    ::puts(no_time_to_die.c_str());
    return EXIT_SUCCESS;
}
