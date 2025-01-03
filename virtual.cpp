#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>

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

template<class _Ty> requires std::is_arithmetic_v<_Ty> class simple_wrapper final {
    private:
        _Ty _wrapped_value;

    public:
        explicit constexpr simple_wrapper(const _Ty& init) noexcept : _wrapped_value(init) { }

        virtual void greet() const noexcept { ::_putws(L"Hi from simple_wrapper"); }

        // return the address of the virtual function table
        uintptr_t vptr() const noexcept { return *reinterpret_cast<const uintptr_t*>(this); }

        _Ty value() const noexcept { return _wrapped_value; }
};

auto wmain() -> int {
    ::srand(::time(nullptr));

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

    std::vector<::simple_wrapper<double>> collection;
    for (const auto& _ : std::ranges::views::iota(0, 100)) collection.emplace_back(::rand() / static_cast<double>(RAND_MAX));

    std::wcout << std::boolalpha;
    const auto vptr = collection.at(0).vptr();
    std::wcout << std::hex << std::uppercase << vptr << L'\n';

    std::wcout << std::all_of(collection.cbegin(), collection.cend(), [&vptr](const auto& wrapped) noexcept -> bool {
        return wrapped.vptr() == vptr;
    }) << L'\n';

    // all should be identical
    for (unsigned i = 0; i < 100; ++i) std::wcout << *reinterpret_cast<uintptr_t*>(&collection.at(i)) << L'\n';
    // print out the random values
    for (unsigned i = 0; i < 100; ++i) std::wcout << *(reinterpret_cast<double*>(&collection.at(i)) + 1) << L'\n';

    std::wcout << L'\n' << collection.back().value() << L'\n';

    return EXIT_SUCCESS;
}
