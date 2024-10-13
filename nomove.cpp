// a C++03 compatible source
#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <vector>

// clang-format off

int main() {
    const ::sstring        Anoban("Anoban");    // char literal ctor
    std::vector< ::sstring> names;
   // names.push_back(Anoban);         // copy ctor
   // names.push_back(Anoban + " :)"); // operator+(), copy ctor
    names.push_back("Gonzalez");       // char literal ctor, copy ctor
    names.push_back(Anoban);    // copy ctor

   // for (std::vector< ::sstring>::const_iterator it = names.begin(), end = names.end(); it != end; ++it) std::cout << *it << '\n';

   // Anoban + " yeehaw \n";

    return EXIT_SUCCESS;
}

// clang-format on
