#include <cstdio>

#if !(__cplusplus > 202002L) // #elifxxx directives are only available with C++23
    #pragma message("Using the handrolled #elifdef variant !")
    #define elifdef(...) elif defined(__VAR_ARGS__)
#endif

auto main() -> int {
#ifdef __clang__
    ::puts("LLVM based compiler\n");
#elifdef(_MSC_FULL_VER)
    ::puts( "Visual C++ compiler\n");
#elifdef(__GNUG__)
    ::puts("GNU g++ compiler\n");
#endif

    return 0;
}
