#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <algorithm>
#include <cstring>
#include <sstring>
#include <type_traits>

static inline void capitalize(_Inout_ ::sstring& string) noexcept { // use of a mutable reference as an in-out function argument
    std::transform(string.begin(), string.end(), string.begin(), ::toupper);
}

static inline void print(_In_ const ::sstring& string) noexcept { ::printf_s("%s\n", string.c_str()); }

static inline ::sstring get() noexcept { return ::sstring("perfect forwarding"); }

auto main() -> int {
    ::sstring name { "Anoban" };
    ::capitalize(name);
    ::print(name);

    ::sstring laptop { "Alienware" };
    static_cast<::sstring&&>(laptop) = "Thinkpad"; // this is valid C++ because
    auto&& xvalue { ::get() };
    xvalue = "still perfect :)"; // needs to be valid since these two are semantically equivalent

    auto* ptr { &::get() };                           // error because we cannot take the address of an unmaterialized temporary (prvalue)
                                                      // or rvalues in general
    auto* _ptr { &static_cast<::sstring&&>(laptop) }; // error for the same reason because std::move here is just static_cast<::sstring&&>()

    return EXIT_SUCCESS;
}
