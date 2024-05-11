// g++ increment.cpp -Wall -Wextra -Wpedantic -std=c++20 -O3

// increment and decrement operator overloading

#ifdef __GNUC__
    #include <cstdio>
#elif defined(__llvm__) || defined(__clang__) || defined(_MSC_VER) || defined(_MSC_FULL_VER) // clang uses MSVC's STL so, :)
    #include <corecrt_wstdio.h>
#endif
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <type_traits>

template<typename T> struct is_character;

template<> struct is_character<char> {
        using type                  = char;
        static constexpr bool value = true;
};

template<> struct is_character<char8_t> {
        using type                  = char8_t;
        static constexpr bool value = true;
};

template<> struct is_character<wchar_t> {
        using type                  = wchar_t;
        static constexpr bool value = true;
};

template<> struct is_character<char16_t> {
        using type                  = char16_t;
        static constexpr bool value = true;
};

template<> struct is_character<char32_t> {
        using type                  = char32_t;
        static constexpr bool value = true;
};

template<typename char_t>
constexpr inline typename std::enable_if<::is_character<char_t>::value, char_t>::type newline = static_cast<char_t>('\n');

template<typename scalar_t, typename = std::enable_if<std::is_scalar_v<scalar_t>, scalar_t>::type> struct number {
        using value_type = scalar_t;

    private:
        value_type value {};

    public:
        constexpr number() noexcept = default;
        constexpr explicit number(const scalar_t v) noexcept : value { v } { }

        constexpr number(const number& other) noexcept : value { other.value } { }
        constexpr number& operator=(const number& other) noexcept {
            value = other.value;
            return *this;
        }

        constexpr number(number&& other)           = delete;
        constexpr number operator=(number&& other) = delete;

        constexpr ~number() noexcept               = default;

        // prefix++
        constexpr number& operator++() noexcept {
            value++;
            return *this;
        }

        // prefix--
        constexpr number& operator--() noexcept {
            value--;
            return *this;
        }

        // postfix++
        constexpr number operator++(int) noexcept {
            value++;
            return number { value - 1 };
        }

        // postfix--
        constexpr number operator--(int) noexcept {
            value--;
            return number { value + 1 };
        }

        constexpr value_type operator()() const noexcept { return value; }

        template<typename char_t> friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const number& num) {
            ostream << num.value << ::newline<char_t>;
            return ostream;
        }

        constexpr void reset() noexcept { value = static_cast<scalar_t>(0); }
};

enum class days : uint8_t { Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday };

enum class months : uint64_t { January, February, March, April, May, June, July, August, September, October, November, December };

enum ten : uint16_t { One = 1, Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten };

int main() {
    auto nineteen { ::number(9) };
    // std::wcout << nineteen;

    for (const auto& _ : std::ranges::views::iota(0, 15)) std::wcout << nineteen--;
    ::_putws(L"");
    for (const auto& _ : std::ranges::views::iota(0, 25)) std::wcout << nineteen++;
    ::_putws(L"");

    (++nineteen).reset();

    for (const auto& _ : std::ranges::views::iota(0, 25)) std::wcout << --nineteen;
    ::_putws(L"");
    for (const auto& _ : std::ranges::views::iota(0, 25)) std::wcout << ++nineteen;
    ::_putws(L"");

    [[maybe_unused]] constexpr auto nl32 { ::newline<char32_t> };
    static_assert(sizeof nl32 == 4);

    return EXIT_SUCCESS;
}
