#include <cstdio>
#include <cstdlib>

auto wmain() -> int {
#if defined(_MSC_VER) && defined(_MSC_FULL_VER) // MSVC or LLVM or Intel
    ::_putws(L"" __FUNCSIG__);
#elif defined(__GNUG__) && defined(__GNUC__)
    ::_putws(L"" __PRETTY_FUNCTION__); // will be a compile time syntax error
#endif

    // __PRETTY_FUNCTION__ is a variable
    // something like const char* const __PRETTY_FUNCTION__ = "some name";
    // so ::_putws(L"" __PRETTY_FUNCTION__); is syntax error

    const char* const __VARIABLE__ { "This is a variable NOT A MACRO!" };
    ::_putws(L"" __VARIABLE__); // syntax error
    // this syntax will work only when __VARIABLE__ is a string literal defined using a macro

#define STRING_LITERAL "STRING LITERAL"
    ::_putws(L"" STRING_LITERAL); // that's cool

    return EXIT_SUCCESS;
}
