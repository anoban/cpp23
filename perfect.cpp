// #define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <type_traits>

namespace experimental {

    // when the caller specifies _Ty to be an lvalue reference, the first overload
    // will be used IF CONST, LVALUE REFERENCES CAN BIND TO RVALUES
    // NOLINTNEXTLINE(modernize-use-constraints)
    template<typename _Ty> static typename std::enable_if<std::is_lvalue_reference<_Ty>::value, _Ty>::type forward(
        typename std::remove_reference<_Ty>::type& _lvref
    ) noexcept {
        ::puts(__FUNCSIG__);
        return _lvref;
    }

    // when the caller specifies _Ty to be a value type (non-reference), the second
    // overload will be used RVALUE REFERENCES CAN NEVER BIND TO LVALUES
    template<typename _Ty>
    static typename std::add_rvalue_reference<_Ty>::type forward(typename std::remove_reference<_Ty>::type&& _rvref) noexcept {
        ::puts(__FUNCSIG__);
        return static_cast<typename std::add_rvalue_reference<_Ty>::type>(_rvref);
    }

} // namespace experimental

template<typename _Ty> // NOLINTNEXTLINE(modernize-use-constraints)
static typename std::enable_if<std::constructible_from<::sstring, _Ty>, void>::type function(_Ty&& _univref) noexcept {
    // when an rvalue is passed, _Ty becomes [cv] ::sstring
    // when an lvalue is passed, _Ty becomes [cv] ::sstring&
    // inside ::function(), _univref is always an lvalue
    [[maybe_unused]] ::sstring temporary(::experimental::forward<_Ty>(_univref));
}

static void print_sstring([[maybe_unused]] ::sstring& lvref) noexcept { ::puts(__FUNCSIG__); }

static void print_sstring([[maybe_unused]] ::sstring&& rvref) noexcept { ::puts(__FUNCSIG__); }

template<typename _Ty> // NOLINTNEXTLINE(modernize-use-constraints)
static typename std::enable_if<std::constructible_from<::sstring, _Ty>, void>::type printer(_Ty&& _univref) noexcept {
    // when an rvalue is passed, _Ty becomes [cv] ::sstring
    // when an lvalue is passed, _Ty becomes [cv] ::sstring&
    // inside ::function(), _univref is always an lvalue!!!!
    ::print_sstring(::experimental::forward<_Ty>(_univref));
}

int main() {
    // example 01 - must invoke the copy ctor
    ::sstring me { "Anoban" };
    ::function(me);

    // example 02 - should call the copy ctor
    ::function(const_cast<const ::sstring&>(me));

    // example 03 - should invoke the move ctor
    // ::function(std::move(me));

    // example 04 - should invoke the move ctor0
    // ::function(::sstring { "Hello there!" });

    // example 05 - should invoke the const char (&)[] ctor
    ::function("string literal");

    // example 06 - should invoke the copy ctor
    const ::sstring what { "const" };
    // ::function(std::move(what));

    ::printer("literal"); // rvalue reference overload
    ::sstring key { "shift" };
    ::printer(key);
    ::printer(::sstring { "prvalue" });

    return EXIT_SUCCESS;
}
