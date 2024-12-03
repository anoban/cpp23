// clang .\enums.cpp -Wall -O3 -std=c++20 -Wextra -pedantic

#include <iostream>

// now, let's try overloading the ++ operator using references
enum class days { MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY };

template<typename T> concept is_character_v = std::is_same_v<char, T> || std::is_same_v<wchar_t, T>;

template<typename T> requires is_character_v<T> static std::basic_ostream<T>& operator<<(std::basic_ostream<T>& ostr, const days& day) {
    if constexpr (std::is_same_v<char, T>) {
        switch (day) {
            case days::MONDAY    : ostr << "MONDAY\n"; break;
            case days::TUESDAY   : ostr << "TUESDAY\n"; break;
            case days::WEDNESDAY : ostr << "WEDNESDAY\n"; break;
            case days::THURSDAY  : ostr << "THURSDAY\n"; break;
            case days::FRIDAY    : ostr << "FRIDAY\n"; break;
            case days::SATURDAY  : ostr << "SATURDAY\n"; break;
            case days::SUNDAY    : ostr << "SUNDAY\n"; break;
        }
    } else if constexpr (std::is_same_v<wchar_t, T>) {
        switch (day) {
            case days::MONDAY    : ostr << L"MONDAY\n"; break;
            case days::TUESDAY   : ostr << L"TUESDAY\n"; break;
            case days::WEDNESDAY : ostr << L"WEDNESDAY\n"; break;
            case days::THURSDAY  : ostr << L"THURSDAY\n"; break;
            case days::FRIDAY    : ostr << L"FRIDAY\n"; break;
            case days::SATURDAY  : ostr << L"SATURDAY\n"; break;
            case days::SUNDAY    : ostr << L"SUNDAY\n"; break;
        }
    }
    return ostr;
}

// implementing a wraparound prefix ++ operator
days operator++(days& rhs) noexcept {
    // if the day is sunday, wrap around to monday
    rhs = (rhs == days::SUNDAY) ? rhs = days::MONDAY : static_cast<days>(static_cast<unsigned>(rhs) + 1);
    return rhs;
}

// in C++, a const reference parameter will accept both const and non-const objects, they both will be treated as constant objects inside the
// function that received them as const references.
static constexpr int addconsts(const int& a /* const reference */, const int& b /* const reference */) noexcept { return a + b; }

static constexpr int add(int& a /* mutable reference */, int& b /* mutable reference */) noexcept { return a + b; }

int                  main() {
    auto day { days::THURSDAY };
    for (unsigned i = 0; i < 14; ++i) std::wcout << ++day; // works as expected

    // as a collateral benefit of using references, prefix ++ operator will only work on lvalue references
    // ++days::MONDAY; Error: this operation on an enumerated type requires an applicable user-defined operator function
    // because our overload only supports days& (lvalue references not prvalues)

    auto       x { 10 }, y { 11 };
    const auto a { 65 }, b { 76 };

    // functions that take const references will happily accept regular references
    addconsts(a, b);
    addconsts(a, x);
    addconsts(x, y);

    add(x, y); // okay
    add(a, x); // Error: qualifiers dropped in binding reference of type "int &" to initializer of type "const int"
    // a const int gets passed as a regular reference, whereby enabling mutation of the passed integer

    return EXIT_SUCCESS;
}
