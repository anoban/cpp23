#include <cctype>
#include <cstdlib>

auto main() -> int {
    //
    constexpr auto newline { '\n' };
    constexpr auto newlineW { L'\n' };
    constexpr auto newlineU8 { u8'\n' };
    constexpr auto newlineU16 { u'\n' };
    constexpr auto newlineU32 { U'\n' };
    constexpr auto newlineR { 'R(\n)' };

    return EXIT_SUCCESS;
}
