#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 1 // NOLINT(cppcoreguidelines-macro-usage)
#include <sstring>

auto main() -> int {
    const ::sstring hello { "Hello" };
    const ::sstring world { "world" };

    std::cout << hello + " " + world << '\n';

    // we cannot take the address of a temporary
    // const ::sstring* const error { &(hello + " there!") };

    // but we can indeed extend the temporary's lifetime by binding it to a const ::sstring& type or ::string&& type or const ::string&&
    const ::sstring&  materialized { hello + " there!" };
    ::sstring&&       temporary { hello + " there!, howdy?" };
    const ::sstring&& simpson { "stupid " + world };

    std::cout << simpson << '\n';

    return EXIT_SUCCESS;
}
