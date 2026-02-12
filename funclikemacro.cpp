#include <cstdio>
#include <cstdlib>

#define greet() ::puts("Hello there!")

#define cube(x) ((x) * (x) * (x))

auto wmain() -> int {
    greet();
    const float sixtyfour = cube(3.0000 + 1);

#undef greet // we do not need to include the parenthesis for undefining function like macros
#undef cube  // no need to use cube()

    return EXIT_SUCCESS;
}
