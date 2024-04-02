// clang .\lambdas.cpp -Wall -Wextra -O3 -std=c++23 -o .\lambdas.exe

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <vector>

// a simple freestanding function
static inline void increment(int&& x) { x++; }

// an equivalent functor
struct incrementor {
        inline void operator()(int&& x) const { x++; }
};

// stateful functor
struct decrementor {
    private:
        std::size_t ninvocations {};

    public:
        static std::size_t ncalls; // for non-const static member variables, the declaration is done inside the class type.
        // but the definition, which grants its storage needs to be done outside the class type.
        constexpr decrementor(void) noexcept { }

        inline void operator()(int&& x) {
            ninvocations++;
            ncalls++;
            x--;
        }

        inline std::size_t count() const noexcept { return ninvocations; }
};

std::size_t decrementor::ncalls {}; // static members can only be defined inside the namespaces or globally
// inside functions, struct::static_member is an expression not a declaration
// in the above definition, removing the storage class type (std::size_t) will make it an expression

int         main(void) {
    //  std::size_t          functor::ncalls = 87; error

    std::vector<int32_t> integrals(100);
    std::iota(integrals.begin(), integrals.end(), 1);

    std::for_each(integrals.cbegin(), integrals.cend(), increment /* raw function pointer */);

    std::for_each(integrals.cbegin(), integrals.cend(), incrementor {} /* function like object (functor) */);

    void (*fnptr)(int&&) = increment;
    (*fnptr)(100);

    auto subtract { decrementor {} };
    // invoking the callable object directly
    subtract.operator()(20);

    const auto USED_FTOR = std::for_each(integrals.cbegin(), integrals.cend(), statefulprinter);
    std::wcout << L"\nstatefulprinter has been invoked " << statefulprinter.count() << L" times!\n";
    std::wcout << L"\nfunctor::ncalls = " << decrementor::ncalls << L" \n";

    std::wcout << L"\nUSED_FTOR has been invoked " << USED_FTOR.count() << L" times!\n";

    // functor passed in as argument to std::for_each doesn't get it's state altered in-place.
    // instead, std::for_each returns a functor object that has the realized state transformations.

    // in cases where the functor is constructed in the call to std::for_each, the modofied state of the functor cannot be affected in-place
    // as the functor gets passed as a prvalue.
    // use the returned functor instead!.
    const auto FT = std::for_each(integrals.cbegin(), integrals.cend(), decrementor {} /* inline construction */);
    std::wcout << L"\nFT has been invoked " << FT.count() << L" times!\n";
    std::wcout << L"\functor::ncalls = " << decrementor::ncalls << L" \n";

    return EXIT_SUCCESS;
}
