// clang .\enum.cpp -Wall -Wextra -O3 -std=c++20 -pedantic

#include <cassert>
#include <concepts>
#include <iostream>

enum class DAY { MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY };

static constexpr const char* const    DAYS[] { "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY", nullptr };
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

template<typename T> requires std::integral<T> && std::is_unsigned<T>::value constexpr DAY operator+(const DAY& day, const T integer) {
    return static_cast<DAY>(day + integer);
}

template<typename T> requires std::integral<T> && std::is_unsigned<T>::value constexpr DAY operator-(const DAY& day, const T integer) {
    return static_cast<DAY>(day - integer);
}

template<typename T> concept is_printable = std::is_same_v<T, char> || std::is_same_v<T, wchar_t>;

template<typename T> requires ::is_printable<T> std::basic_ostream<T>& operator<<(std::basic_ostream<T>& bostr, const DAY day) {
    if constexpr (std::is_same_v<T, char>)
        bostr << DAYS[static_cast<int>(day)] << '\n';
    else
        bostr << WDAYS[static_cast<int>(day)] << L'\n';
    return bostr;
}

static constexpr void testOperators() noexcept {
    auto prefix { DAY::MONDAY };
    auto postfix { DAY::SUNDAY };

    for (unsigned i = 0; i < 28; ++i) {
        assert(++prefix == DAY::MONDAY + (i % 7));
        assert(postfix++ == DAY::SUNDAY + (i % 7) - 1);
    }
}
auto main() -> int {
    auto day { DAY::MONDAY };

    std::wcout << day++; // MONDAY

    std::wcout << ++day; // WEDNESDAY

    std::wcout << --day; // TUESDAY

    std::wcout << day--; // TUESDAY

    std::wcout << day; // MONDAY

    testOperators();

    return EXIT_SUCCESS;
}
