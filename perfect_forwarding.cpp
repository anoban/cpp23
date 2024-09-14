#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 1
#include <sstring> // love it :)
#include <type_traits>

// slightly doctored versions of STL's std::forward implementations
namespace nstd {

    // consider template<class T> explicit user(const T&& init) noexcept : name(std::forward<T>(init)) { ::puts(__FUNCSIG__); }
    // forwarding references can bind to lvalue references, const lvalue references, rvalue references and const rvalue references!

    template<typename T> [[nodiscard]] constexpr T&& forward(typename std::remove_reference_t<T>& _Arg) noexcept {
        return static_cast<T&&>(_Arg);
    }

    template<typename T> requires std::is_lvalue_reference_v<T>
    [[nodiscard]] constexpr T&& forward(typename std::remove_reference_t<T>&& _Arg) noexcept {
        // static_assert(!std::is_lvalue_reference_v<T>, "bad forward call");
        return static_cast<T&&>(_Arg);
    }

} // namespace nstd

static inline void function(_In_ [[maybe_unused]] const ::sstring& string) noexcept { ::puts(__FUNCSIG__); }

static inline void function(_In_ [[maybe_unused]] ::sstring&& string) noexcept { ::puts(__FUNCSIG__); }

namespace rvalues {
    static inline void function(_In_ [[maybe_unused]] ::sstring&& string) noexcept { ::puts(__FUNCSIG__); }
} // namespace rvalues

namespace lvalues {
    static inline void function(_In_ [[maybe_unused]] const ::sstring& string) noexcept { ::puts(__FUNCSIG__); }
} // namespace lvalues

class user final {
        ::sstring name;

    public:
        template<class T> explicit user(const T& init) noexcept : name(init) { ::puts(__FUNCSIG__); }
        // explicit user(::sstring&& _name) noexcept : name(std::move(_name)) /* move construction */ { ::puts(__FUNCSIG__); }
        // explicit user(::sstring&& _name) noexcept : name(_name) /* copy construction */ { ::puts(__FUNCSIG__); }

        user()                       = delete;
        user(const user&)            = delete;
        user(user&&)                 = delete;
        user& operator=(const user&) = delete;
        user& operator=(user&&)      = delete;

        ~user() noexcept             = default;
};

class book final {
        ::sstring title;
        ::sstring author;

    public:
        template<class T, class U>
        explicit book(T&& _title /* forwarding reference */, U&& _author /* forwarding reference */) noexcept :
            title { nstd::forward<T>(_title) }, author { nstd::forward<U>(_author) } {
            ::puts(__FUNCSIG__);
        }

        book()                       = delete;
        book(const book&)            = delete;
        book(book&&)                 = delete;
        book& operator=(const book&) = delete;
        book& operator=(book&&)      = delete;

        ~book() noexcept             = default;
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

    lvalues::function("Anoban"); // prvalue temporaries can bind to const lvalue references
    lvalues::function(adele);    // okay

    user            natalie { "Natalie" /* implicit conversion to ::sstring */ };
    book            prisoner_of_azkaban { "Harry Potter and the Prisoner of Azkaban", "J.K.Rowling" };
    const ::sstring rowling { "J.K.Rowling" };
    ::sstring       chamber { "Harry Potter and the Chamber of Secrets" };
    ::sstring       philosopher { "Harry Potter and the Philosopher's Stone" };

    book philosophers_stone { philosopher, rowling };        // expect both to be copy constructed
    book chamber_of_secrets { std::move(chamber), rowling }; // title will be move constructed, author will be copy constructed
    return EXIT_SUCCESS;
}
