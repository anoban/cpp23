#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <cstdlib>
#include <numbers>
#include <sstring>

namespace overloads {

    // inline void function([[maybe_unused]] ::sstring str) noexcept { ::puts(__FUNCSIG__); }
    inline void function([[maybe_unused]] ::sstring& str) noexcept { ::puts(__FUNCSIG__); }
    inline void function([[maybe_unused]] const ::sstring& str) noexcept { ::puts(__FUNCSIG__); }
    inline void function([[maybe_unused]] ::sstring&& str) noexcept { ::puts(__FUNCSIG__); }
    inline void function([[maybe_unused]] const ::sstring&& str) noexcept { ::puts(__FUNCSIG__); }

} // namespace overloads

namespace recursion {
    [[nodiscard]] static ::sstring&& append(_Inout_ ::sstring&& string, _In_ ::sstring&& annexure, _In_ const unsigned times) noexcept {
        string += annexure;
        if (!times) return std::move(string);
        return append(std::move(string), std::move(annexure), times - 1);
    }
} // namespace recursion

auto main() -> int {
    const ::sstring immutable { "what is hapening?" };
    ::sstring       modifiable { "this is const" };

    overloads::function(immutable);  // should call the const lvalue reference overload
    overloads::function(modifiable); // should call the non-const lvalue reference overload

    overloads::function("converting ctor"); // should call the rvalue reference overload
    // clang and g++ are okay with this but MSVC overload resolution errs saying this is an ambiguous call

    overloads::function(std::move(modifiable)); // should call the rvalue reference overload
    overloads::function(std::move(immutable));  // should call the const rvalue reference overload

    return EXIT_SUCCESS;
}
