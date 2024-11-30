#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <type_traits>

namespace experimental {

    // when the caller specifies _Ty to be an lvalue reference, the first overload will be used
    // IF CONST, LVALUE REFERENCES CAN BIND TO RVALUES
    template<typename _Ty> static _Ty forward(typename std::remove_reference<_Ty>::type& _lvref) noexcept {
        ::puts(__FUNCSIG__);
        return _lvref;
    }

    // when the caller specifies _Ty to be a value type (non-reference), the second overload will be used
    // RVALUE REFERENCES CAN NEVER BIND TO LVALUES
    template<typename _Ty>
    static typename std::add_rvalue_reference<_Ty>::type forward(typename std::remove_reference<_Ty>::type&& _rvref) noexcept {
        ::puts(__FUNCSIG__);
        return std::move(_rvref);
    }

} // namespace experimental

template<typename _Ty>
static typename std::enable_if<std::constructible_from<::sstring, _Ty>, void>::type function(_Ty&& _univref) noexcept {
    [[maybe_unused]] ::sstring temporary(::experimental::forward<_Ty>(_univref));
}

int main() {
    // example 01 - must invoke the copy ctor
    ::sstring me { "Anoban" };
    ::function(me);

    // example 02 - should also call the copy ctor
    ::function(const_cast<const ::sstring&>(me));

    // example 03 - should invoke the move ctor
    ::function(std::move(me));

    return EXIT_SUCCESS;
}
