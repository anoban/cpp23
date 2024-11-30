#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <type_traits>

template<class _Ty //, class = typename std::enable_if<std::is_same<typename std::remove_cvref<_Ty>::type, ::sstring>::value, _Ty>::type
         >
inline void perfect_forwarder(_Ty&& string /* ::string types */
) noexcept(std::is_nothrow_copy_constructible_v<typename std::remove_cvref_t<_Ty>> && std::is_nothrow_move_constructible_v<typename std::remove_cvref_t<_Ty>>) {
    // if string is a prvalue temporary, T will be deduced as ::sstring so T&& can be ::sstring&&
    // if string is an lvalue, T will be deduced as ::sstring& so T&& can be ::sstring&&
    [[maybe_unused]] ::sstring copy { std::forward<_Ty>(string) };
}

// without perfect forwarding
template<class _Ty //, class = typename std::enable_if<std::is_same<typename std::remove_cvref<_Ty>::type, ::sstring>::value, _Ty>::type
         >
inline void forwarder(_Ty&& string /* ::string types */
) noexcept(std::is_nothrow_copy_constructible_v<typename std::remove_cvref_t<_Ty>> && std::is_nothrow_move_constructible_v<typename std::remove_cvref_t<_Ty>>) {
    // if string is a prvalue temporary, T will be deduced as ::sstring so T&& can be ::sstring&&
    // if string is an lvalue, T will be deduced as ::sstring& so T&& can be ::sstring&&
    [[maybe_unused]] ::sstring copy { string }; // will always invoke the copy ctor
}

auto main() -> int {
    ::puts("--------------------------------------------------------------------------------");

    ::sstring adele { "skyfall" };
    ::perfect_forwarder("string literal");                  // should call the string literal ctor
    ::perfect_forwarder(::sstring { "prvalue temporary" }); // should call the move ctor
    ::perfect_forwarder(adele);                             // should call the copy ctor
    ::perfect_forwarder(std::move(adele));                  // should call the move ctor

    ::puts("--------------------------------------------------------------------------------");
    ::puts("--------------------------------------------------------------------------------");

    ::sstring billie { "no time to die" };
    ::forwarder("string literal");                  // should call the string literal ctor
    ::forwarder(::sstring { "prvalue temporary" }); // should call the copy ctor
    ::forwarder(billie);                            // should call the copy ctor
    ::forwarder(std::move(billie));                 // should call the copy ctor

    ::puts("--------------------------------------------------------------------------------");
    return EXIT_SUCCESS;
}
