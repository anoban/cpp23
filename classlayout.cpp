// cl.exe /std:c++20 /d1reportSingleClassLayout<class name>
// cl.exe /std:c++20 /d1reportAllClassLayout
// clang -emit-llvm -std=c++20 -Xclang -fdump-record-layouts -Xclang -fdump-vtable-layouts -c

// for MSVC, existence of class definition is enough to dump the layouts
// for LLVM, the class must at least be instantiated once in the programme

#include <cstdio>
#include <cstdlib>

class stationary {
    protected:
        float    _price { 1000.0 };
        float    _profmar { 12.35 };
        unsigned _stock { 7'134 };

    public:
        virtual void           huh() const noexcept { _putws(L"virtual void stationary::huh() const noexcept"); }

        virtual const wchar_t* name() const noexcept { return L"stationary"; }
};

static_assert(sizeof(stationary) == 24);

/*
    layout of class stationary ::
    0000 vtableptr
    0008 _price
    0012 _profmar
    0016 _stock
    <4 padding bytes>
*/

class mimic_stationary {
        uintptr_t _padd0 {};
        uintptr_t _padd1 {};

    public:
        virtual void           placeholder() const noexcept { _putws(L"virtual void mimic_stationary::placeholder() const noexcept"); }

        virtual const wchar_t* what() const noexcept { return L"mimic_stationary"; }
};

class exercise_book : public stationary {
    protected:
        unsigned _pages { 240 };

    public:
        virtual void           huh() const noexcept override { _putws(L"virtual void exercise_book::huh() const noexcept override"); }

        virtual const wchar_t* name() const noexcept override { return L"exercise_book"; }
};

static_assert(sizeof(mimic_stationary) == sizeof(stationary));

auto wmain() -> int {
    constexpr exercise_book ebook {}; // to make clang AST dumping work

    ebook.huh();
    dynamic_cast<const stationary*>(&ebook)->huh();

    constexpr mimic_stationary dummy {};
    reinterpret_cast<const stationary*>(&dummy)->huh();          // should invoke placeholder()
    _putws(reinterpret_cast<const stationary*>(&dummy)->name()); // should invoke what()

    // this type of type punning works well with clang, msvc, icx and g++ :)

    return EXIT_SUCCESS;
}

// IF YOU WANT CRUDE BINARY LEVEL REIMAGINING OF CLASS USE REINTERPRET CAST NOT DYNAMIC CAST!
