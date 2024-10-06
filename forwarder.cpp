#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <type_traits>

template<class T, class = typename std::enable_if<std::is_same<typename std::remove_cvref<T>::type, ::sstring>::value, T>::type>
static inline void perfect_forwarder(T&& string /* ::string types */) noexcept {
    // if string is a prvalue temporary, T will be deduced as ::sstring so T&& can be ::sstring&&
    // if string is an lvalue, T will be deduced as ::sstring& so T&& can be ::sstring&&
    ::sstring copy { std::forward<T>(string) };
    std::cout << copy << '\n';
}

auto main() -> int {
    ::sstring adele { "skyfall" };
    ::perfect_forwarder(::sstring("prvalue temporary")); // should call the move ctor
    ::perfect_forwarder(adele);                          // should call the copy ctor
    ::perfect_forwarder(std::move(adele));               // should call the move ctor
    return EXIT_SUCCESS;
}
