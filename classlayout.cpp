// cl.exe /std:c++20 /d1reportSingleClassLayout<class name>
// cl.exe /std:c++20 /d1reportAllClassLayout
// clang -emit-llvm -std=c++20 -Xclang -fdump-record-layouts -Xclang -fdump-vtable-layouts -c

// for MSVC existence of class definition is enough to dump the layouts
// for clang, the class must at least be used once in the programme

#include <cstdio>
#include <cstdlib>

class stationary {
    protected:
        float    _price;
        float    _profmar;
        unsigned _stock;

    public:
        virtual void how_much() const noexcept { _putws(L"virtual void stationary::how_much() const noexcept"); }
};

class exercise_book : public stationary {
    protected:
        unsigned _pages;

    public:
        virtual void how_much() const noexcept override { _putws(L"virtual float exercise_book::how_much() const noexcept override"); }
};

auto wmain() -> int {
    constexpr exercise_book ebook {}; // to make clang AST dumping work

    dynamic_cast<const stationary*>(&ebook)->how_much();
    ebook.how_much();

    return EXIT_SUCCESS;
}
