#include <cctype>
#include <concepts>
#include <cstdlib>
#include <numbers>

template<typename T> consteval T                                   newline() noexcept { return T('\n'); }

template<typename T> inline constexpr T                            nl                  = T('\n');

template<std::floating_point _Floating> inline constexpr _Floating log10e_v<_Floating> = static_cast<_Floating>(0.4342944819032518);

auto                                                               main() -> int {
    //
    constexpr auto newline { '\n' };
    constexpr auto newlineW { L'\n' };
    constexpr auto newlineU8 { u8'\n' };
    constexpr auto newlineU16 { u'\n' };
    constexpr auto newlineU32 { U'\n' };
    constexpr auto newlineR { 'R(\n)' };

    const auto     guess { ::nl<char32_t> };

    return EXIT_SUCCESS;
}
