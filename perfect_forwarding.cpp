#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 1
#include <sstring> // love it :)

static inline void function(_In_ [[maybe_unused]] const ::sstring& string) noexcept {
    ::puts(__FUNCSIG__);
    ::puts(string.c_str());
}

static inline void function(_In_ [[maybe_unused]] ::sstring&& string) noexcept {
    ::puts(__FUNCSIG__);
    ::puts(string.c_str());
}

namespace rvalues {
    static inline void function(_In_ [[maybe_unused]] ::sstring&& string) noexcept {
        ::puts(__FUNCSIG__);
        ::puts(string.c_str());
    }
} // namespace rvalues

namespace const_lvalues {
    static inline void function(_In_ [[maybe_unused]] const ::sstring& string) noexcept {
        ::puts(__FUNCSIG__);
        ::puts(string.c_str());
    }
} // namespace const_lvalues

class user final {
        ::sstring name;

    public:
        template<class T> explicit user(T&& /* forwarding reference */ init) noexcept : name(init) { ::puts(__FUNCSIG__); }

        user()                       = delete;
        user(const user&)            = delete;
        user(user&&)                 = delete;
        user& operator=(const user&) = delete;
        user& operator=(user&&)      = delete;

        ~user() noexcept             = default;
};

auto main() -> int {
    // string literals are implicitly converted to ::sstring, results in a temporary materialization
    ::function("Anoban"); // uses the rvalue reference overload
    ::sstring adele { "I set fireeeeee to the rain!" };
    ::function(adele);                        // uses the const lvalue reference overload
    ::function(std::move_if_noexcept(adele)); // uses the rvalue reference overload

    // we didn't do anything to the string `adele` inside ::function so it should still be usable
    ::puts(adele.c_str()); // and it is!

    rvalues::function("Anoban");         // okay, the materialized ::sstring temporary will bind as an rvalue reference
                                         //  rvalues::function(adele);            // error, an lvalue cannot bind to an rvalue reference
    rvalues::function(std::move(adele)); // okay, an xvalue can bind to an rvalue reference

    const_lvalues::function("Anoban"); // prvalue temporaries can bind to const lvalue references
    const_lvalues::function(adele);    // okay

    user natalie { "Natalie" };

    return EXIT_SUCCESS;
}
