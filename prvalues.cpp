// there's a distinction between xvalues and prvalues!!
#include <numbers>
#include <string>

// values returned by functions are prvalues even if they are of class types
[[nodiscard]] std::wstring static inline helloooooooooooo_from_the_other_sideeeeeeeeeeeeeee(
) noexcept(std::is_nothrow_constructible_v<std::wstring>) {
    return L"I might have called a thousand tiiiiiiiiiiiiimessssssssssssss!";
}
// the wstring object returned by helloooooooooooo_from_the_other_sideeeeeeeeeeeeeee is a prvalue NOT AN XVALUE

static void seize(std::wstring&& str, const unsigned& recursion_count) throw() {
    // str can be an xvalue or rvalue at the call site
    // inside the function body, it becomes an lvalue because it is bound to an identifier
    auto* _addr { &str }; // okay because str is an lvalue inside this function body

    if (!recursion_count) return;
    ::_putws(str.c_str());
    ::seize(std::move(str), recursion_count - 1); // seize() only accepts rvalue references so we need to re promote our lvalue string to
    // an rvalue (xvalue)
    // moves are not propagated automatically!
}

template<typename T> struct pair {
        using value_type = T;

        T first;
        T second;
};

#define NAME L"ANOBAN"
#define AGE  45LL

enum sex : int16_t { MALE, FEMALE, NONBINARY };

auto wmain() -> int {
    auto           adele { ::helloooooooooooo_from_the_other_sideeeeeeeeeeeeeee() };
    constexpr auto ten {
        // CTAD is in the works!
        ::pair { 10, 12 }
         .first  // this member access will not be possible without the returned struct occupying storage
    };

    auto&& ref { ::helloooooooooooo_from_the_other_sideeeeeeeeeeeeeee() };
    ::_putws(ref.c_str());
    ::_putws(adele.c_str());

    MALE++;
    AGE++;
    ++AGE;

    auto pi { std::numbers::pi_v<double> };
    auto egamma { std::numbers::egamma };
    pi *= 10;

    auto&       refpi { pi };
    auto* const ptrpi { &pi };
    // the above constant pointer and reference are functionally similar
    refpi  -= 10;
    *ptrpi += 20;

    ptrpi   = &egamma; // NOPE
    refpi   = egamma;  // writes through the reference, DOES NOT REALIASE THE REFERENCE TO egamma!

    const auto& crefpi { pi }; // const reference to pi even though pi itself is not a const object!
    // this reference has only read privileges
    auto        x  = crefpi - 100;
    crefpi        *= 12; // NOPE

    const auto* const cptrpi { &pi };
    x       += *cptrpi;
    *cptrpi *= 11; // UH HUH

    {
        // consider this scope
        const std::wstring billy { L"Was it obvious to everybody else?" };
        std::wstring       eilish { L"that I've fallen for a lie" };

        ::seize(std::move(eilish), 10); // std::move promotes the lvalue to an xvalue
        ::seize(billy, 10);             // error
    }

    return EXIT_SUCCESS;
}

// prvalues  - pure rvalues
// include literals except string literals and function return values, unnamed lambdas and temporaries
// prvalues do not have a capturable memory address, even class type prvalues
// xvalues are lvalues explicitly casted to rvalues using a static_cast<T&&>() or std::move()

static void function() noexcept(false) {
    const auto* const lvptr = &L"I've beeeeen here beforeeeeeeeeeeee!";                      // okay, reference to an lvalue
    auto&             isit  = std::wstring { L"I feel like a storm is coming!" };            // mhm, cannot take a reference to a prvalue
    const auto&       may = std::wstring { L"because the writing's on the waaaaallllllll" }; // prvalues can bind to const lvalue references

    const float& willit   = std::numbers::pi_v<float>;
}
