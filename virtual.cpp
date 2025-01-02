#include <iostream>

class base {
    public:
        unsigned _value;

        virtual constexpr const wchar_t* const str() const noexcept { return L"base"; }

        void greet() const noexcept { ::_putws(L"Hi from base!"); }
};

class derived : public base {
    public:
        double pay;

        constexpr const wchar_t* const str() const noexcept override { return L"derived"; }

        void greet() const noexcept { ::_putws(L"Hi from derived!"); }
};

struct last : derived {
        constexpr const wchar_t* const str() const noexcept override { return L"last"; }
};

auto wmain() -> int {
    const derived object {};

    std::wcout << object.str() << '\n';                             // derived
    std::wcout << static_cast<const base*>(&object)->str() << '\n'; // derived

    object.greet();                             // Hi from derived!
    static_cast<const base*>(&object)->greet(); // Hi from base!

    const last example {};
    std::wcout << static_cast<const base*>(&example)->str() << '\n';

    return EXIT_SUCCESS;
}
