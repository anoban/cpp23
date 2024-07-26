#include <cassert>
#include <cstdlib>

enum class month { January, February, March, April, May, June, July, August, September, October, November, December };

constexpr month& operator++(month& now) noexcept {
    if (now == month::December)
        now = month::January;
    else
        now = static_cast<month>(static_cast<int>(now) + 1);
    return now;
}

constexpr month operator++(month& now, int) noexcept {
    const auto temp { now };
    if (now == month::December)
        now = month::January;
    else
        now = static_cast<month>(static_cast<int>(now) + 1);
    return temp;
}

template<size_t n> struct factorial {
        static constexpr size_t value { n * factorial<n - 1>::value };
};

template<> struct factorial<0> {
        static constexpr size_t value { 1 };
};

static_assert(factorial<5>::value == 120);
static_assert(factorial<6>::value == 720);
static_assert(factorial<0>::value == 1);
static_assert(factorial<1>::value == 1);

// variadic templates can be instantiated without template arguments too
template<class... TList> struct user { };
user<unsigned, char, wchar_t, unsigned long, const float, const volatile double>
    julia {}; // templated struct user instantiated with a parameter pack

user<> empty {}; // okay to instantiate a class type template with an empty template argument list

template<class... TList> static constexpr unsigned func(const TList&... arg_pack); // just the declaration

func<>(); // won't work because to deduce TList the compiler needs an argument pack or an explicit type list
// since none of that are here, this will not be interpreted as a template instantiation at all
// clang sees this as an attempt to define a template specialization with the template<> part missing in the front

template<class... TList_0, class... TList_1> constexpr void function() noexcept { }

func<unsigned, const float&, volatile double&&>();

template<typename... TList> [[nodiscard]] consteval double sum(const TList&... args) noexcept { return (args + ... + 0); }

template<typename... TList> [[nodiscard]] consteval double mul(const TList&... args) noexcept { return (... * args); }

namespace using_overloads {

    template<class T, class... TList> [[nodiscard]] consteval double sum(const T& start, const TList&... rest) throw() {
        return start + sum(rest...);
    }

    template<class T> [[nodiscard]] consteval double sum(const T& end) throw() { return end; }

    template<class T, class... TList> [[nodiscard]] consteval double mul(const T& start, const TList&... rest) throw() {
        return start + mul(rest...);
    }

    template<class T> [[nodiscard]] consteval double mul(const T& end) throw() { return end; }
} // namespace using_overloads

namespace using_constexpr_if {
    template<class T, class... TList> [[nodiscard]] consteval double sum(const T& start, const TList&... rest) throw() {
        if constexpr (sizeof...(TList) == 1) return rest;
        return start + sum(rest...);
    }

    template<class T, class... TList> [[nodiscard]] consteval double mul(const T& start, const TList&... rest) throw() {
        if constexpr (sizeof...(TList) == 0) return start;
        return start * mul(rest...);
    }
} // namespace using_constexpr_if

static_assert(::sum(1, 2.0, 3.00L, 4u, 5l, 6ll) == 21);
static_assert(::mul(1.00f, 2llu, 3zu, 4ui16, 5i8, 6i16) == 720);
static_assert(::mul(0, 1, 2, 3, 4, 5, 6) == 0);

static_assert(using_constexpr_if::sum(1, 2.0, 3.00L, 4u, 5l, 6ll) == 21);
static_assert(using_constexpr_if::mul(1.00f, 2llu, 3zu, 4ui16, 5i8, 6i16) == 720);
static_assert(using_constexpr_if::mul(0, 1, 2, 3, 4, 5, 6) == 0);

static void constexpr test_month_increment_operators() throw() {
    auto today { month::July };
    for (unsigned i = static_cast<unsigned>(today); i < 100; i = ++i % 12) {
        // assert(static_cast<unsigned>(today++) == i - 1);

        assert(static_cast<unsigned>(++today) == i);
    }
}

auto wmain() -> int {
    test_month_increment_operators();
    return EXIT_SUCCESS;
}
