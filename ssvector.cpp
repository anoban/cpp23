#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <vector>

int main() {
    std::vector<sstring> collect;
    collect.emplace_back("Hello there!"); // std::vector::emplace_back is a C++11 feature
    for (std::vector<sstring>::const_iterator it = collect.begin(), end = collect.end(); it != end; ++it) std::cout << *it << '\n';
    return EXIT_SUCCESS;
}
