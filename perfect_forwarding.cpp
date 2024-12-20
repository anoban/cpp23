#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
// #define __SSTRING_NO_MOVE_SEMANTICS__
#include <sstring>
#include <type_traits>
#include <vector>

// slightly doctored versions of STL's std::forward implementations
namespace nstd {

    // consider template<class T> explicit user(const T&& init) noexcept : name(std::forward<T>(init)) { ::puts(__FUNCSIG__); }
    // forwarding references can bind to lvalue references, const lvalue references, rvalue references and const rvalue references!
    // in perfect forwarding the T of std::forward<T>() comes from the outer template where T&& was a universal reference

    // there is no type deduction in std::forward<T>, the type deduction happens in the outer template and the deduced
    // template paramater is used to explicitly instantiate std::forward<T>

    // in case of an rvalue reference T will just be a type e.g. if T&& is std::string&& then T is std::string
    // in case of an lvalue reference T will a reference type e.g. if T&& is std::string& then T will be std::string&
    // because T& + && = T&& (reference collapsing)

    template<typename _Ty> [[nodiscard]] constexpr _Ty&& forward(typename std::remove_reference_t<_Ty>&
                                                                     _Arg /* _Arg will only bind with lvalue references */) noexcept {
        return static_cast<_Ty&&>(_Arg); // T& + && -> T&, return type is an lvalue reference
    }

    template<typename _Ty> requires(!std::is_lvalue_reference_v<_Ty>) // this overload will only be used when _Arg is an rvalue reference
    // and T is deduced to be a non-reference type
    [[nodiscard]] constexpr _Ty&& forward(typename std::remove_reference_t<_Ty>&& _Arg /* _Arg will only bind with rvalue references */
    ) noexcept {
        // static_assert(!std::is_lvalue_reference_v<T>, "bad forward call"); refactored this into a requires clause
        return static_cast<_Ty&&>(_Arg); // T&& + && -> T&&, return type is an rvalue reference
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
        template<class _Ty> explicit user(const _Ty& init) noexcept : name(init) { ::puts(__FUNCSIG__); }

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
        template<class _Ty, class U> explicit book(_Ty&& _title /* forwarding reference */, U&& _author /* forwarding reference */) noexcept
            :
            // if T is deduced to be ::sstring (which will happen when _title is an rvalue reference ::sstring&&)
            // nstd::forward<T>(_title) will use the second overload with a rvalue reference argument
            title { nstd::forward<_Ty>(_title) }, author { nstd::forward<U>(_author) } {
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
    rvalues::function(adele);            // error, an rvalue reference cannot be bound to an lvalue

    lvalues::function("Anoban"); // prvalue temporaries can bind to const lvalue references
    lvalues::function(adele);    // okay

    user            natalie { "Natalie" /* implicit conversion to ::sstring */ };
    book            prisoner_of_azkaban { "Harry Potter and the Prisoner of Azkaban", "J.K.Rowling" };
    const ::sstring rowling { "J.K.Rowling" };
    ::sstring       chamber { "Harry Potter and the Chamber of Secrets" };
    ::sstring       philosopher { "Harry Potter and the Philosopher's Stone" };

    book philosophers_stone { philosopher, rowling }; // expect both to be copy constructed
    // book chamber_of_secrets { std::move(chamber), rowling }; // title will be move constructed, HADN'T WE DELETED THE MOVE CTOR
    book chamber_of_secrets { chamber, rowling };

    ::puts(".................................");

    std::vector<::sstring> container(100);
    container.push_back("rvalue");
    container.emplace_back("rvalue");

    container.push_back(rowling);
    container.emplace_back(rowling);
    return EXIT_SUCCESS;
}
