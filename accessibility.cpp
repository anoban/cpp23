#include <iostream>
#include <type_traits>

#pragma pack(push, 1)
class bclass {
    public:
        short           _pushort { 11 };
        constexpr short getpus() const noexcept { return _pushort; }

    protected:
        short           _prshort { 22 };
        constexpr short _getprs() const noexcept { return _prshort; }

    private:
        short           _pvshort { 33 };
        constexpr short _getpvs() const noexcept { return _pvshort; }
};
#pragma pack(pop)

static_assert(!std::is_standard_layout_v<bclass>); // WOW, so this is not a POD class
static_assert(sizeof(bclass) == 6);                // no padding bytes huh!

class dclass : public bclass {
    public:
        constexpr short getprs() const noexcept { return _getprs(); }
        short           getpvs() const noexcept { return *(reinterpret_cast<const short*>(this) + 2); } // crude and delicate but works :)
};

auto wmain() -> int {
    constexpr auto derived { dclass {} };

    std::wcout << derived.getpus() << L'\n';
    std::wcout << derived.getprs() << L'\n';
    // we call the derived class's public getter to access the base class's protected member via the base class's protected member function
    std::wcout << derived.getpvs() << L'\n';

    return EXIT_SUCCESS;
}
