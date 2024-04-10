#include <concepts>
#include <iostream>

template<typename T> requires std::integral<T> class book {
    public:
        T nborrows {};

        book() = delete;

        explicit constexpr book(T n) noexcept : nborrows { n } { }

        constexpr book& operator++() noexcept { // ++prefix
            nborrows++;
            return *this;
        }

        constexpr book& operator--() noexcept { // --prefix
            nborrows--;
            return *this;
        }

        // with postfixed operators this mutate the internal state and return a new object having the old state idiom is
        // used to replicate the behaviour of postfixed ++ & -- operators with primitive types.
        constexpr book operator++(int) noexcept { // postfix++
            nborrows++;
            return book { nborrows - 1 };         // return the older state
        }

        constexpr book operator--(int) noexcept { // postfix--
            nborrows--;
            return book { nborrows + 1 };
        }
};

template<typename T> concept is_cout_compatible = std::is_same_v<T, char> || std::is_same_v<T, wchar_t>;

template<typename T> requires ::is_cout_compatible<T> constexpr T nl() noexcept {
    if constexpr (std::is_same_v<T, char>) return '\n';
    return L'\n';
}

template<typename char_t, typename integral_t> requires std::integral<integral_t> && ::is_cout_compatible<char_t>
std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostr, const book<integral_t>& object) {
    ostr << object.nborrows << ::nl<char_t>();
    return ostr;
}

template<typename char_t, typename integral_t> requires std::integral<integral_t> && ::is_cout_compatible<char_t>
std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostr, const book<integral_t>&& object) {
    ostr << object.nborrows << ::nl<char_t>();
    return ostr;
}

int main() {
    long x { 45 };
    auto HarryPotterAndTheChamberOfSecrets { book { 565LL } };

    // in C, both prefix and postfix ++ & -- operators return values
    // after all C does not have references

    // prefix ++ and -- operators return a reference
    ::wprintf_s(L"%ld\n", ++x);
    ::wprintf_s(L"%ld\n", x);
    --x = 0; // assignment is possible here as --x returns a reference to x
    ::wprintf_s(L"%ld\n", x);

    // postfix ++ and -- operators return a value
    // x++ = 0; Error: expression must be a modifiable lvalue
    // assignment is an error here because x++ returns a prvalue, which cannot be assigned to

    std::wcout << HarryPotterAndTheChamberOfSecrets;

    std::wcout << HarryPotterAndTheChamberOfSecrets++; // 565
    std::wcout << HarryPotterAndTheChamberOfSecrets--; // 566
    std::wcout << HarryPotterAndTheChamberOfSecrets;   // 565

    std::wcout << ++HarryPotterAndTheChamberOfSecrets; // 566
    std::wcout << --HarryPotterAndTheChamberOfSecrets; // 565
    std::wcout << HarryPotterAndTheChamberOfSecrets;   // 565

    return 0;
}
