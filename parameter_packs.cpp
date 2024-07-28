#include <cassert>
#include <cstdlib>

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

template<class... TList> static constexpr unsigned func(const TList&... arg_pack) noexcept; // just the declaration

// func<>(); // won't work because to deduce TList the compiler needs an argument pack or an explicit type list
// since none of that are here, this will not be interpreted as a template instantiation at all
// clang sees this as an attempt to define a template specialization with the template<> part missing in the front

template<class... TList_0, class... TList_1> constexpr void function() noexcept { }

// func<unsigned, const float&, volatile double&&>();

const auto result { func() };
const auto _result { func(47, 5.012, 0.765743f) };

namespace using_fold_expressions {
    template<typename... TList> [[nodiscard]] consteval double sum(const TList&... args) noexcept {
        return (args + ... + 0); // NOLINT(cppcoreguidelines-narrowing-conversions)
    }

    template<typename... TList> [[nodiscard]] consteval double mul(const TList&... args) noexcept {
        return (... * args); // NOLINT(cppcoreguidelines-narrowing-conversions)
    }
} // namespace using_fold_expressions

namespace using_overloads {

    // the order of the declaration of these two template overloads matter
    // the overload taking a single scalar must precede the overload with the argument pack
    template<class T> [[nodiscard]] consteval double sum(const T& val) throw() { return val; }

    template<class T, class... TList> [[nodiscard]] consteval double sum(const T& start, const TList&... rest) throw() {
        return start + using_overloads::sum(rest...); // NOLINT(cppcoreguidelines-narrowing-conversions)
    }

    template<class T> [[nodiscard]] consteval double mul(const T& val) throw() { return val; }

    template<class T, class... TList> [[nodiscard]] consteval double mul(const T& head, const TList&... rest) throw() {
        return head * using_overloads::mul(rest...); // NOLINT(cppcoreguidelines-narrowing-conversions)
    }

} // namespace using_overloads

namespace using_constexpr_if {

    // with a consteval function, we know for a fact that the function will be evaluated at compile time
    // but if statements must still be explicitly constexpered if we want them to be evaluated @ compiletime

    template<class T, class... TList> [[nodiscard]] consteval double sum(const T& head, const TList&... pack) throw() {
        if constexpr (sizeof...(TList) == 0)
            return head; // NOLINT(cppcoreguidelines-narrowing-conversions)
        else
            return head + using_constexpr_if::sum(pack...); // NOLINT(cppcoreguidelines-narrowing-conversions)
    }

    template<class T, class... TList> [[nodiscard]] consteval double mul(const T& head, const TList&... pack) throw() {
        if constexpr (sizeof...(TList) == 0)
            return head;
        else
            return head * using_constexpr_if::mul(pack...); // NOLINT(cppcoreguidelines-narrowing-conversions)
    }
} // namespace using_constexpr_if

static_assert(using_fold_expressions::sum(1, 2.0, 3.00L, 4u, 5l, 6ll, '\n') == 31);
static_assert(using_fold_expressions::sum('A', 1, 2.0, 3.00L, 4u, 5l, 6ll, '\n') == 96);
static_assert(using_fold_expressions::mul(1.00f, 2llu, 3ll, 4ui16, 5i8, 6i16) == 720);
static_assert(using_fold_expressions::mul(0, 1, 2, 3, 4, 5, 6) == 0);

static_assert(using_constexpr_if::sum(1, 2.0, 3.00L, 4u, 5l, 6ll) == 21);
static_assert(using_constexpr_if::sum('A', 1, 2.0, 3.00L, 4u, 5l, 6ll, '\n') == 96);
static_assert(using_constexpr_if::mul(1.00f, 2llu, 3ll, 4ui16, 5i8, 6i16) == 720);
static_assert(using_constexpr_if::mul(0, 1, 2, 3, 4, 5, 6) == 0);

static_assert(using_overloads::sum(1, 2.0, 3.00L, 4u, 5l, 6ll, '\t') == 30);
static_assert(using_overloads::sum('A', 1, 2.0, 3.00L, 4u, 5l, 6ll, '\n') == 96);
static_assert(using_overloads::mul(1.00f, 2llu, 3ll, 4ui16, 5i8, 6i16) == 720);
static_assert(using_overloads::mul(0, 1, 2, 3, 4, 5, 6) == 0);

auto wmain() -> int { return EXIT_SUCCESS; }
