// return value optimizations
#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <cctype>
#include <sstring>

// RVO
static ::sstring get() noexcept { return "Hi there!"; } // MSVC, LLVM and ICX optimized this into a RVO assisted in-place construction

// NRVO
static ::sstring nget() noexcept { // MSVC, LLVM and ICX optimized this into a NRVO assisted in-place construction
    ::sstring named { "Hi there!" };
    return named;
}

static ::sstring cget() noexcept {
    const ::sstring named { "Hi there!" }; // the const here may disable NRVO
    return named;                          // will probably be copy constructed at call site
    // BUT MSVC, LLVM and ICX optimized this into a NRVO assisted in-place construction
}

static ::sstring ccget() noexcept { // MSVC, LLVM and ICX optimized this into a NRVO assisted in-place construction WTF?????
    ::sstring named { "Hi there!" };
    for (auto& character : named) character = static_cast<char>(::toupper(character));
    return named;
}

auto wmain() -> int {
    const auto howmany { ::get() }; // without RVO, this should print two ctor invocations when called
    // one ctor call with a string literal as argument
    // followed by a move ctor call
    // with RVO, the ::sstring will be constructed in-place using the string literal

    const auto nowwhat { ::nget() }; // without NRVO, this should call the string literal ctor first and the move ctor next
    // with NRVO this will only call the string literal ctor to construct the ::sstring in-place

    const auto ohwell { ::cget() }; // NRVO works here too

    const auto nada { ::ccget() }; // NRVO
    std::wcout << nada << L'\n';

    std::wcout << std::endl;

    return EXIT_SUCCESS;
}
