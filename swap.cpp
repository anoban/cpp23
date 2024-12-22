#include <vector>
#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 1 // NOLINT
#include <sstring>

static void _swap(_Inout_ ::sstring& left, _Inout_ ::sstring& right) noexcept {
    ::sstring temporary { left }; // copy construction
    left  = right;                // copy assignment
    right = temporary;            // copy assignment
}

static void swap(_Inout_::sstring& left, _Inout_::sstring& right) noexcept {
    ::sstring temporary { std::move(left) }; // move construction
    left  = std::move(right);                // move assignment
    right = std::move(temporary);            // move assignment
}

template<typename _Ty>
std::ostream& operator<<(_Inout_ std::ostream& ostr, _In_ const std::vector<_Ty>& vect) noexcept(noexcept(std::cout << vect.at(0)))
    requires requires { ostr << vect.at(0); } {
    ostr << "{ ";
    for (unsigned long long i = 0; i < vect.size() - 1; ++i) ostr << vect.at(i) << ", ";
    ostr << vect.at(vect.size() - 1) << " }\n";
    return ostr;
}

auto main() -> int {
    ::sstring hello { "world!\n" };
    ::sstring world { "Hello " };

    ::swap(hello, world);
    std::cout << hello << world;

    hello.swap(world);
    std::cout << hello << world;

    ::_swap(hello, world);
    std::cout << hello << world;

    std::vector<double> first { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<double> second { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    std::cout << first;
    std::cout << second;

    first.swap(second);

    std::cout << first;
    std::cout << second;

    return EXIT_SUCCESS;
}
