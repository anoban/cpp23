// clang .\enum.cpp -Wall -Wextra -std=c++20 -O3 -pedantic -DDEBUG  -o enum.exe

#include <cassert>
#include <concepts>
#include <iostream>

constexpr std::size_t TEST_MAX_ITERS { 49 }; // loop through this many times!

enum class DAY { MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY };

static constexpr const char* const DAYS[] {
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY", nullptr
}; // ending with a nullptr in MS style

static constexpr const wchar_t* const WDAYS[] {
    L"MONDAY", L"TUESDAY", L"WEDNESDAY", L"THURSDAY", L"FRIDAY", L"SATURDAY", L"SUNDAY", nullptr
};

constexpr DAY& operator++(DAY& day) noexcept { // prefix++
    day = (day == DAY::SUNDAY) ? DAY::MONDAY : static_cast<DAY>(static_cast<int>(day) + 1);
    return day;
}

constexpr DAY& operator--(DAY& day) noexcept { // prefix--
    day = (day == DAY::MONDAY) ? DAY::SUNDAY : static_cast<DAY>(static_cast<int>(day) - 1);
    return day;
}

constexpr DAY operator++(DAY& day, int) noexcept { // postfix++
    if (day == DAY::SUNDAY) {
        day = DAY::MONDAY;
        return DAY::SUNDAY;
    } else {
        day = static_cast<DAY>(static_cast<int>(day) + 1);
        return static_cast<DAY>(static_cast<int>(day) - 1);
    }
}

constexpr DAY operator--(DAY& day, int) noexcept { // postfix--
    if (day == DAY::MONDAY) {
        day = DAY::SUNDAY;
        return DAY::MONDAY;
    } else {
        day = static_cast<DAY>(static_cast<int>(day) - 1);
        return static_cast<DAY>(static_cast<int>(day) + 1);
    }
}

template<typename T> concept is_printable = std::is_same_v<T, char> || std::is_same_v<T, wchar_t>;

template<typename T> requires ::is_printable<T> std::basic_ostream<T>& operator<<(std::basic_ostream<T>& bostr, const DAY day) {
    if constexpr (std::is_same_v<T, char>)
        bostr << DAYS[static_cast<int>(day)] << '\n';
    else
        bostr << WDAYS[static_cast<int>(day)] << L'\n';
    return bostr;
}

// works :)))
static constexpr void TestIncrementOperators() noexcept {
    auto prefix { DAY::MONDAY };
    auto postfix { DAY::SUNDAY };

    for (unsigned i = 1; i <= TEST_MAX_ITERS; ++i) {
        assert(++prefix == static_cast<DAY>(i % 7)); // DAY::MONDAY is 0!
        // after five ++prefix calls prefix will be 5
        assert(postfix++ == static_cast<DAY>((i + 5) % 7)); // DAY::SUNDAY is 6!, hence the 6 - 1 = 5!
    }
}

// works too :)))
static constexpr void TestDecrementOperators() noexcept {
    auto     prefix { DAY::THURSDAY };
    auto     postfix { DAY::TUESDAY };

    unsigned pre { 3 /* DAY::THRUSDAY */ }, post { 1 /* DAY::TUESDAY */ };

    for (unsigned i = 1; i <= TEST_MAX_ITERS; ++i) {
        pre = (pre) ? pre - 1 : 6; // --prefix will do the decrement first and then the return, so this needs to precede that statement
                                   // for correct evaluation
        assert(--prefix == static_cast<DAY>(pre));

        assert(postfix-- == static_cast<DAY>(post)); // postfix-- will return the decremented value and will then materialize the decrement
        // internally, thus the decrement of post follows that statement!
        post = (post) ? post - 1 : 6;
    }
}

auto main() -> int {
    auto day { DAY::MONDAY };

    std::wcout << day++; // MONDAY

    std::wcout << ++day; // WEDNESDAY

    std::wcout << --day; // TUESDAY

    std::wcout << day--; // TUESDAY

    std::wcout << day; // MONDAY

    TestIncrementOperators();
    TestDecrementOperators();

    return EXIT_SUCCESS;
}
