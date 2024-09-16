#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 1
#include <sstring>

static inline void _swap(_Inout_ ::sstring& left, _Inout_ ::sstring& right) noexcept {
    ::sstring temporary { left }; // copy construction
    left  = right;                // copy assignment
    right = temporary;            // copy assignment
}

static inline void swap(_Inout_::sstring& left, _Inout_::sstring& right) noexcept {
    ::sstring temporary { std::move(left) }; // move construction
    left  = std::move(right);                // move assignment
    right = std::move(temporary);            // move assignment
}

auto main() -> int {
    ::sstring hello { "world!\n" };
    ::sstring world { "Hello " };
    ::swap(hello, world);

    std::cout << hello << world;

    ::_swap(hello, world);
    std::cout << hello << world;
    return EXIT_SUCCESS;
}
