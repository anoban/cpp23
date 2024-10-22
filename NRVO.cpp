#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <array>
#include <iostream>
#include <sstring>
#include <string>

// named return value optimizations (NRVO) require a valid move ctor

struct copy_only final {
        explicit copy_only(float x) noexcept : data(x) { }

        copy_only(const copy_only&)            = default;
        copy_only& operator=(const copy_only&) = default;

        copy_only(copy_only&&)                 = delete;
        copy_only& operator=(copy_only&&)      = delete;

        float data;
};

static inline copy_only nrvo() noexcept {
    const copy_only welp { 0.0 };
    return welp; // okay because no move semantics for const objects
}

static inline copy_only nrvo_failure() noexcept {
    copy_only welp { 0.0 };
    return welp; // call to a deleted move ctor
}

static inline copy_only rvo() noexcept { // this is not NAMED RETURN VALUE OPTIMIZATION but just regular RETURN VALUE OPTIMIZATION
    return copy_only { 0.0 };
    // this will work even with both copy and move ctors deleted
}

class none final { // not copyable and not movable
    private:
        double data;

    public:
        none() noexcept : data() { }
        none(const double& v) noexcept : data(v) { }
        none(const none&)            = delete;
        none(none&&)                 = delete;
        none& operator=(const none&) = delete;
        none& operator=(none&&)      = delete;
        ~none() noexcept             = default;
};

static none _rvo() noexcept { return none { 54.0 }; } // no valid copy ctor or move ctor but this will still work :)
// plain old RVO

static none nrvo_err() noexcept {
    const none nada { 0.354789 };
    return nada; // error, deleted copy ctor
}

static none nrvo_err_too() noexcept {
    none nada { 0.354789 };
    return nada; // error, deleted move ctor
}

int main() {
    std::cout << std::uppercase;

    std::array<::sstring, 10> strings { "Jane", "James", "Janelle", "Juliana", "Jamie", "Joseph", "Jacob", "Janice", "Jack" };
    ::sstring                 hispanic { "Juan" };
    // *strings.end() = hispanic; // one-off error, .end() is one element past the last!!!!
    strings.at(strings.size() - 1) = hispanic;         // lvalue - copy assignment
    strings.at(strings.size() - 1) = hispanic + "ita"; // prvalue - move assignment

    for (std::array<::sstring, 10>::const_iterator it = strings.cbegin(), end = strings.cend(); it != end; ++it) std::cout << *it << '\n';

    // move ctor
    ::sstring juanita { std::move(strings.at(9)) };
    std::cout << juanita << '\n';

    // should be NULL
    std::cout << "strings.at(9).c_str() is  " << (strings.at(9).c_str() ? "not NULL" : "NULL") << '\n';

    juanita = static_cast<::sstring&&>(strings.at(2)); // move assignment
    std::cout << juanita << '\n';                      // Janelle

    const auto monet { juanita + " monet" }; // move ctor

    // C++ standard does not gurantee that moved from objects will be NULL or NULL equivalent
    std::string       jeremey { "Jeremey Fischer god damn make this long enough so SSO won't work here" };
    const std::string moved { static_cast<std::string&&>(jeremey
    ) }; // after move construction, jeremey's internal buffer did not get NULL ed here
    // the move ctor leaves the moved from object in a default constructed state!!!!!
    std::cout << "jeremey.c_str() is " << (jeremey.c_str() ? "not NULL" : "NULL") << '\n';

    std::cout << "size " << jeremey.size() << " " << moved.size() << '\n';
    std::cout << "capacity " << jeremey.capacity() << " " << moved.capacity() << '\n';
    std::cout << "data " << reinterpret_cast<ptrdiff_t>(jeremey.c_str()) << " " << reinterpret_cast<ptrdiff_t>(moved.c_str()) << '\n';

    std::string default_constructed {};
    std::cout << "size " << default_constructed.size() << '\n';
    std::cout << "capacity " << default_constructed.capacity() << '\n';
    std::cout << "data " << reinterpret_cast<ptrdiff_t>(default_constructed.c_str()) << '\n';

    return EXIT_SUCCESS;
}

// retrun value optimization & named return value optimizations are completely at the compiler's discretion
// if RVO or NRVO does not happen, the compiler may fall back to using copy or move semantics
// moves are preferred over copies
