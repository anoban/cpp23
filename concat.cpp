#include <cstdio>
#include <cstdlib>

auto wmain() -> int {
#if defined(_MSC_VER) && defined(_MSC_FULL_VER) // MSVC or LLVM or Intel
    ::_putws(L"" __FUNCSIG__);
#elif defined(__GNUG__) && defined(__GNUC__)
    ::_putws(L"" __PRETTY_FUNCTION__);
#endif
    return EXIT_SUCCESS;
}
