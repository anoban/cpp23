#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <type_traits>

namespace nstd {

    template<class _Ty> struct is_lvalue_reference final {
            static constexpr bool value = false;
    };

    template<class _Ty> struct is_lvalue_reference<_Ty&> final {
            using type                  = _Ty;
            static constexpr bool value = true;
    };

    static_assert(!is_lvalue_reference<double>::value);
    static_assert(!is_lvalue_reference<const double>::value);
    static_assert(!is_lvalue_reference<const double&&>::value);
    static_assert(is_lvalue_reference<const double&>::value);
    static_assert(is_lvalue_reference<volatile float&>::value);
    static_assert(!is_lvalue_reference<const double*>::value);

    // this overload can never bind to rvalue references (because it takes a regular reference to lvalues)
    template<class _Ty> [[nodiscard]] constexpr _Ty&& forward(typename std::remove_reference<_Ty>::type& _arg) noexcept {
        // this overload is used when arg is an lvalue reference
        // in the outer template, to make the universal reference _Ty&& an lvalue reference type, _Ty will be deduced as an lvalue reference type _Ty& (reference collapsing)
        return static_cast<_Ty&&>(_arg); // _Ty& + && -> _Ty& (we return an lvalue reference)
    }

    template<class _Ty> requires(!is_lvalue_reference<_Ty>::value) // this overload can bind to both lvalue and rvalue references
    // hence the type constraint
    [[nodiscard]] constexpr _Ty&& forward(typename std::remove_reference<_Ty>::type&& _arg) noexcept {
        // since the outer template signature is _Ty&& (a universal reference) it will only bind to references
        // when arg is an rvalue reference i.e _Ty&&, according to reference collapsing _Ty will be deduced as a plain value type in the outer template
        return static_cast<_Ty&&>(_arg);
        // we cast the rvalue reference (which has now become an lvalue) to an rvalue reference and return it
        // and our return type _Ty&& will become an rvalue reference, so okay
    }

    // STD::FORWARD DOES NOT DO TYPE DEDUCTION AND REQUIRES THE CALLER TO EXPLICITLY SPECIFY THE TEMPLATE TYPE ARGUMENT

} // namespace nstd

namespace fwd {
    // overload for lvalue references
    template<typename _Ty> requires std::is_lvalue_reference_v<_Ty>
    [[nodiscard]] constexpr _Ty forward(_In_ typename std::remove_reference<_Ty>::type& arg) noexcept {
        return arg; // return the lvalue reference
    }
    // overload for rvalue references
    template<typename _Ty> [[nodiscard]] constexpr _Ty&& forward(_In_ typename std::remove_reference<_Ty>::type&& arg) noexcept {
        return std::move(arg); // equivalent to return static_cast<_Ty&&>(arg);
    }
} // namespace fwd

template<class _Ty> static void forwarder(_Ty&& arg /* a universal reference in the outer template */) noexcept(noexcept(::sstring(arg))) {
    const ::sstring dummy { nstd::forward<_Ty>(arg) };
}

template<typename _Ty> requires std::constructible_from<::sstring, _Ty> static void forwarding(_Ty&& univref) noexcept {
    const ::sstring temporary { fwd::forward<_Ty>(univref) };
    ::puts(temporary.c_str());
}

auto main() -> int {
    ::forwarder("string literal"); // receives a char array reference as argument
    // deduced type _Ty by ::func will be const char (&)[15]
    // inside first overload of nstd::forward this will be the type of _arg, which will return the same type

    ::forwarder(::sstring("prvalue")); // now ::func receives a prvalue temporary ::sstring object
                                       // _Ty deduced by ::func will be ::sstring (a plain value type)
    // inside the second overload of nstd::forward, the type of _arg will be ::sstring&& which will be the return type too
    // this will invoke the move ctor of the ::sstring class

    const ::sstring lvalue("has a name huh!");
    ::forwarder(lvalue); // type _Ty deduced by ::func is ::sstring&
    // the first overload of nstd::forward gets called and the return type is ::sstring& + && = ::sstring&
    // so this will invoke the copy ctor of the ::sstring class

    ::sstring move { "to be moved" };

    ::forwarding("literals");
    // ::forwarding(::sstring { "pure rvalue" });
    ::forwarding(lvalue);
    ::forwarding(move);
    // ::forwarding(std::move(move));

    return EXIT_SUCCESS;
}
