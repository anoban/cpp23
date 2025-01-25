#include <cstdio>

#if !(__cplusplus > 202002L) // #elifxxx directives are only available with C++23
    #pragma message("Using the handrolled #elifdef variant !")
    #define elifdef(...) elif defined(__VAR_ARGS__)
#endif

auto main() -> int {
#ifdef __clang__
    ::_putws(L"LLVM based compiler\n");
#elifdef(_MSC_FULL_VER)
    ::_putws(L "Visual C++ compiler\n");
#elifdef(__GNUG__)
    ::_putws(L"GNU g++ compiler\n");
#endif

    return 0;
}
