#include <iostream>

class base {
    public:
        virtual constexpr const wchar_t* str() const noexcept { return L"base"; }

        void greet() const noexcept { ::_putws(L"Hi from base!"); }
};

class derived : public base {
    public:
        constexpr const wchar_t* str() const noexcept override { return L"derived"; }

        void greet() const noexcept { ::_putws(L"Hi from derived!"); }
};

struct last : derived {
        constexpr const wchar_t* str() const noexcept override { return L"last"; }
};

static_assert(sizeof(last) == 8); // because of the vptr

auto wmain() -> int {
    const derived object {};

    std::wcout << object.str() << '\n';                             // derived
    std::wcout << static_cast<const base*>(&object)->str() << '\n'; // derived

    object.greet();                             // Hi from derived!
    static_cast<const base*>(&object)->greet(); // Hi from base!

    const last example {};
    std::wcout << static_cast<const base*>(&example)->str() << '\n';

    base dummy {};
    std::wcout << static_cast<last*>(&dummy)->str() << '\n'; // downcasting
    // update the vptr of a base class instance to the vptr of the derived class instance
    *reinterpret_cast<uintptr_t*>(&dummy) = *reinterpret_cast<const uintptr_t*>(&object);
    std::wcout << static_cast<last*>(&dummy)->str() << '\n'; // ????

    return EXIT_SUCCESS;
}
