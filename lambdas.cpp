// clang .\lambdas.cpp -Wall -Wextra -O3 -std=c++23 -o .\lambdas.exe

#include <algorithm>
#include <array>
#include <concepts>
#include <iostream>
#include <numeric>
#include <vector>

template<typename T> requires std::is_arithmetic<T>::value
static std::wostream& operator<<(std::wostream& ostr, const std::vector<T>& vector) {
    for (const T& elem : vector) ostr << elem << L' ';
    ostr << L'\n';
    return ostr;
}

template<typename T, size_t size>
// typename = std::enable_if<std::integral<std::remove_cv_t<T>> || std::floating_point<std::remove_cv_t<T>>>::value>
static std::wostream& operator<<(std::wostream& ostr, const std::array<T, size>& array) {
    for (const T& elem : array) ostr << elem << L' ';
    ostr << L'\n';
    return ostr;
}

// a simple freestanding function
static constexpr void increment(int& x) { x++; }

// an equivalent functor
struct incrementor {
        constexpr void operator()(int& x) const { x++; }
};

// stateful functor
struct decrementor {
    private:
        std::size_t ninvocations {};

    public:
        static std::size_t ncalls; // for non-const static member variables, the declaration is done inside the class type.
        // but the definition, which grants its storage needs to be done outside the class type.
        constexpr decrementor() = default;

        void operator()(int& x) noexcept {
            ninvocations++;
            ncalls++;
            x--;
        }

        constexpr std::size_t count() const noexcept { return ninvocations; }
};

std::size_t decrementor::ncalls {}; // static members can only be defined inside the namespaces or globally
// inside functions, struct::static_member is an expression not a declaration
// in the above definition, removing the storage class type (std::size_t) will make it an expression

int         main() {
    auto                 x { 100 };
    std::vector<int32_t> integrals(100);
    std::iota(integrals.begin(), integrals.end(), 1);
    std::wcout << integrals;

    std::for_each(integrals.begin(), integrals.end(), increment /* raw function pointer */);

    auto inc { incrementor {} };
    std::for_each(integrals.begin(), integrals.end(), inc /* function like object (functor) */);
    std::wcout << integrals;

    void (*fnptr)(int&) = increment;
    (*fnptr)(x);

    auto dec { decrementor {} }; // stateful
    // invoking the callable object directly
    dec.operator()(x);
    dec.operator()(x); // decr.count() = 2
    std::wcout << L"decrementor functor dec has been invoked " << dec.count() << L" times!\n";

    std::for_each(integrals.begin(), integrals.end(), dec);
    std::wcout << L"decrementor functor dec has been invoked " << dec.count() << L" times!\n";
    std::wcout << L"decrementor::ncalls = " << decrementor::ncalls << L'\n';
    std::wcout << integrals;
    // functor passed in as argument to std::for_each doesn't get it's state altered in-place.
    // instead, std::for_each returns a functor object that has the realized state transformations.
    // capture the returned functor to materialize the mutated state

    // in cases where the functor is constructed inline in the call to std::for_each, the only way to capture altered state of the functor
    // is to use the returned functor object!
    const auto decr = std::for_each(integrals.begin(), integrals.end(), decrementor {} /* inline construction */);
    std::wcout << L"decrementor object returned by std::for_each has been invoked " << decr.count() << L" times!\n";
    std::wcout << L"decrementor::ncalls = " << decrementor::ncalls << L'\n';
    std::wcout << integrals;
    // or one could assign the returned functor to the passed functor to materialize the modified state, but this will overwrite the existing state
    auto decrm { decrementor {} };
    decrm = std::for_each(integrals.begin(), integrals.end(), decrm);
    std::wcout << L"decrementor object decrm has been invoked " << decrm.count() << L" times!\n";
    std::wcout << L"decrementor::ncalls = " << decrementor::ncalls << L'\n';
    // this approach will become problematic when decrm already had a non-default state, which will get overwritten by the returned functor

    constexpr auto arr { std::array<int16_t, 20> {} };
    std::wcout << arr;

    return EXIT_SUCCESS;
}
