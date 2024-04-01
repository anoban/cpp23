// clang .\lambdas.cpp -Wall -Wextra -O3 -std=c++23 -o .\lambdas.exe

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <vector>

// a simple freestanding function
static inline void print(const int& x) { std::wcout << x << L'\n'; }

// an equivalent functor
struct printer {
        constexpr printer(void) = default;
        inline void operator()(const int& x) const { std::wcout << x << L'\n'; }
};

// stateful functor
struct functor {
    private:
        std::size_t ninvocations {};

    public:
        constexpr functor(void) noexcept { }

        inline void operator()(const int& x) {
            ninvocations++;
            std::wcout << x << L'\n';
        }

        inline std::size_t count() const noexcept { return ninvocations; }
};

int main(void) {
    std::vector<int32_t> integrals(100);
    std::iota(integrals.begin(), integrals.end(), 1);

    std::for_each(integrals.cbegin(), integrals.cend(), print /* raw function pointer */);

    constexpr auto PRINTS { printer {} };
    std::for_each(integrals.cbegin(), integrals.cend(), PRINTS /* function like object (functor) */);

    void (*fnptr)(const int&) = print;
    (*fnptr)(100);

    // invoking the callable object directly
    PRINTS.operator()(20);

    auto       statefulprinter { functor {} };
    const auto USED_FTOR = std::for_each(integrals.cbegin(), integrals.cend(), statefulprinter);
    std::wcout << L"statefulprinter has been invoked " << statefulprinter.count() << L" times!\n";
    std::wcout << L"USED_FTOR has been invoked " << USED_FTOR.count() << L" times!\n";

    return EXIT_SUCCESS;
}
