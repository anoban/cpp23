#include <cstdio>
#include <cstdlib>
#include <type_traits>

template<typename... _TyChar, unsigned long long... _sizes> static void print(const _TyChar (&... _strings)[_sizes]) noexcept {
    // https://stackoverflow.com/questions/53281096/apply-function-to-all-elements-of-parameter-pack-from-a-variadic-function
    (::_putws(_strings), ...); // C++17
}

template<typename... _TyChar, unsigned long long... _sizes> requires(std::is_same_v<_TyChar, wchar_t> && ...)
static void inoculate(_In_ [[maybe_unused]] const unsigned long long& _population_size, _In_ const _TyChar (&... _strings)[_sizes]) {
    (::_putws(_strings), ...);
}

auto wmain() -> int {
    ::print(L"Hi!", L"Hello", L"Howdy!", L"How was your day?");
    ::inoculate(100, L"Hi!", L"Hello", L"Howdy!", L"How was your day?");

    return EXIT_SUCCESS;
}
