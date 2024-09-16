#include <cstdio>
#include <cstdlib>

#define greet() ::_putws(L"Hello there!")

auto wmain() -> int {
    greet();
#undef greet
    return EXIT_SUCCESS;
}
