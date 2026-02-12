#include <cstdio>
#include <cstdlib>

auto wmain() -> int {
#if defined(_MSC_VER) && defined(_MSC_FULL_VER) // MSVC or LLVM or Intel
    ::puts("" __FUNCSIG__);
#elif defined(__GNUG__) && defined(__GNUC__)
    ::puts("" __PRETTY_FUNCTION__); // will be a compile time syntax error
#endif

    // __PRETTY_FUNCTION__ is a variable
    // something like const char* const __PRETTY_FUNCTION__ = "some name";
    // so ::puts("" __PRETTY_FUNCTION__); is syntax error

    const char* const __VARIABLE__ { "This is a variable NOT A MACRO!" };
    ::puts("" __VARIABLE__); // syntax error
    // this syntax will work only when __VARIABLE__ is a string literal defined using a macro

#define STRING_LITERAL "STRING LITERAL"
    ::puts("" STRING_LITERAL); // that's cool

    return EXIT_SUCCESS;
}
