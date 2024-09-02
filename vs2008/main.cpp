#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 1
#include <algorithm>
#include <sstring>

static inline void __stdcall sentence_case(::sstring& str) throw() {
    for (unsigned long i = 0; i < str.length(); ++i)    // my bad :(
        if ((str.at(i) == ' ') && (str.at(i + 1) >= 97) && (str.at(i + 1) <= 122)) str.at(i + 1) -= 32;
}

int main() {
    const ::sstring brennan = "I remember that time we were sippin' on wine";
    std::cout << brennan << '\n';

    ::sstring copy = brennan;
    std::cout << copy << '\n';
    std::transform(copy.cbegin(), copy.cend(), copy.begin(), ::toupper);
    std::cout << copy << '\n';

    ::sstring lithe  = "heard you want a piece of the pie huh?";
    lithe           += " all the smokin' made me tired! Bitch I see the legacy, It's qiet!\n";
    std::cout << lithe;

    ::sentence_case(lithe);
    std::cout << lithe;

    std::cout << "__cplusplus is " << __cplusplus << '\n';
    ::getchar();

    return EXIT_SUCCESS;
}
