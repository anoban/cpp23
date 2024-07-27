// cl .\months.cpp /Wall /WX /Ot /O2 /std:c++20 /DDEBUG /D_DEBUG /EHac /GL /Qpar /MTd

#include <cassert>
#include <cstdio>
#include <cstdlib>

enum class month { January, February, March, April, May, June, July, August, September, October, November, December };

[[nodiscard]] static constexpr month& operator++(month& now) noexcept {
    now = (now == month::December) ? month::January : static_cast<month>(static_cast<int>(now) + 1);
    return now;
}

[[nodiscard]] static constexpr month operator++(month& now, int) noexcept {
    const auto old { now };
    now = (now == month::December) ? month::January : static_cast<month>(static_cast<int>(now) + 1);
    return old;
}

[[nodiscard]] static constexpr month& operator--(month& now) noexcept {
    now = (now == month::January) ? month::December : static_cast<month>(static_cast<int>(now) - 1);
    return now;
}

[[nodiscard]] static constexpr month operator--(month& now, int) noexcept {
    const auto old { now };
    now = (now == month::January) ? month::December : static_cast<month>(static_cast<int>(now) - 1);
    return old;
}

static constexpr auto LIMIT { 48LLU }; // 4 x 12

auto main() -> int {
    auto           july { month::July }, november { month::November };
    constexpr auto _november = static_cast<unsigned>(month::November); // 10
    constexpr auto _july     = static_cast<unsigned>(month::July);     // 6

    for (unsigned i = _july; i < LIMIT; ++i) assert(static_cast<unsigned>(++july) == (i + 1) % 12);
    // cannot just use _july here because static_cast<unsigned>(july) != _july @ this point
    // we need to capture the current state of july or reset july to month::July!
    for (unsigned i = static_cast<unsigned>(july); i < LIMIT; ++i) assert(static_cast<unsigned>(july++) == i % 12);

    for (unsigned i = LIMIT + _november; i > 0; --i) assert(static_cast<unsigned>(--november) == (i - 1) % 12);
    // cannot just use LIMIT + _november here because static_cast<unsigned>(november) != _november @ this point
    // we need to capture the current state of november or reset november to month::November!
    for (unsigned i = LIMIT + static_cast<unsigned>(november); i > 0; --i) assert(static_cast<unsigned>(november--) == i % 12);

    ::_putws(L"all's good:)");

    // assert(false);
    return EXIT_SUCCESS;
}
