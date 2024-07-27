// g++ recursivetemplates.cpp -Wall -Wextra -O3 -std=c++20 -fext-numeric-literals -municode

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define STRINGIFY(MACRO) #MACRO
#define TO_STRING(MACRO) STRINGIFY(MACRO)

// TO_STRING will first expand the macro which will then be stringified by STRINGIFY
// TO_STRING cannot be used with literals, it's argument must be a macro

// LLVM uses MSVC's STL on windows, so _MSC_VER and _MSC_FULL_VER are also defined by LLVM on windows
// that's why the conditional for LLVM must precede that of MSVC
// LLVM and GNU compilers' version macro are strings, so we do not need to stringify them by hand!

#if (defined(__clang__) && defined(__llvm__))
    #pragma message("Compiling " __FILE__ " using LLVM " __clang_version__)
    #define FUNCTION_SIGNATURE __PRETTY_FUNCTION__
#elif (defined(__GNUC__) && defined(__GNUG__))
    #pragma message("Compiling " __FILE__ " using g++ " __VERSION__)
    #define FUNCTION_SIGNATURE __PRETTY_FUNCTION__
#elif defined(_MSC_VER) && defined(_MSC_FULL_VER)
    #pragma message("Compiling " __FILE__ " using MSVC " TO_STRING(_MSC_FULL_VER))
    #define FUNCTION_SIGNATURE __FUNCSIG__
#else
    #error Please find a compiler that provides an extension macro for function signatures!
#endif

template<class T> [[nodiscard]] static constexpr long double sum(const T& tail) throw() {
    ::puts(FUNCTION_SIGNATURE);
    return tail;
}

template<class T, class... TList> [[nodiscard]] static constexpr long double sum(const T& head, const TList&... pack) throw() {
    ::puts(FUNCTION_SIGNATURE);
    return head + ::sum(pack...);
}

auto wmain() -> int {
    [[maybe_unused]] const auto total = ::sum(12.98f, 87, 873L, 0b1101101010, 763U, 'X', L'A');
    return EXIT_SUCCESS;
}
