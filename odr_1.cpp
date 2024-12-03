#include <iostream>

__declspec(noinline) void function() noexcept;

static unsigned           internal_linkage { 82 };

unsigned                  external_linkage { 11 };

extern const double       pie; // odr_1.obj : error LNK2001: unresolved external symbol "double const pie" (?pie@@3NB)

auto                      wmain() -> int {
    std::wcout << pie << L'\n';

    std::wcout << external_linkage << L'\n';
    ::function();
    std::wcout << external_linkage << L'\n';

    return EXIT_SUCCESS;
}
