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
        virtual float how_much() const noexcept { return _price; }
};

class exercise_book : public stationary {
    protected:
        unsigned _pages;

    public:
        virtual float how_much() const noexcept override { return 1000.0; }
};

auto wmain() -> int {
    exercise_book ebook {}; // to make clang AST dumping work
    return EXIT_SUCCESS;
}
