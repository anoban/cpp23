#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <vector>

static const ::sstring get_greeting() noexcept { return "Hi there!"; }

static ::sstring       greeting() noexcept { return "Hello there!"; }

static void            print(::sstring string) noexcept { ::puts(string.c_str()); }

int                    wmain() {
    ::sstring              ano { "Anoban" };
    std::vector<::sstring> strings;

    strings.push_back(get_greeting()); // const ::sstring& overload
    strings.push_back(greeting());     // ::sstring&& overload

    ::print(ano);            // `string` argument of ::print() is copy constructed
    ::print(greeting());     // `string` argument of ::print() may be move constructed unless RVO kicks in
    ::print(std::move(ano)); // `string` argument of ::print() is move constructed

    return EXIT_SUCCESS;
}
