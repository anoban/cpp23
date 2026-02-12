#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

// signature of an array reference is T (&identifier)[length]

static __attribute__((noinline)) double cube(const float& x) noexcept { return x * x * x; }

static __attribute__((noinline)) double quad(const float& x) noexcept { return x * x * x * x; }

template<class T, size_t len> static __attribute__((noinline)) void quad(T (&array)[len]) noexcept {
    for (T& e : array) e *= (e * e * e);
}

template<size_t __len> static inline char* capitalize(char (&string)[__len]) noexcept {
    ::puts(__PRETTY_FUNCTION__);
    for (unsigned i = 0; i < __len; ++i) string[i] = static_cast<char>(::toupper(string[i]));
    return string;
}

static inline char* capitalize(char* string) noexcept {
    ::puts(__PRETTY_FUNCTION__);
    while (*string) *string++ = static_cast<char>(toupper(*string));
    return string;
}

template<class T, size_t __len> static constexpr inline void square(T (&array)[__len]) noexcept requires requires(T& x) { x *= x; } {
    for (size_t i = 0; i < __len; ++i) array[i] *= array[i];
}

template<class _Ty0, class _Ty1, size_t len>
static typename std::enable_if<std::is_arithmetic_v<_Ty0> && std::is_arithmetic_v<_Ty1>, _Ty0 (&)[len]>::type operator+(
    _Ty0 (&left)[len], const _Ty1 (&right)[len]
) noexcept {
    for (size_t i = 0; i < len; ++i) left[i] += right[i];
    return left;
}

auto wmain() -> int {
    double (*const ptrcube)(const float&) noexcept = cube;
    //  ptrcube = quad;

    unsigned array[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    for (const auto& e : array) ::printf("%6u\n", e);
    ::puts("\n");

    square(array);
    for (const auto& e : array) ::printf("%6u\n", e);

    ::printf("%lf\n", (*ptrcube)(3.000));

    char adele[] { "Let the sky fall.............. when it crumbles...." };
    ::capitalize(adele);
    ::printf("%S\n", adele);

    quad(array);
    for (const auto& e : array) ::printf("%6u\n", e);

    // ::printf("%S\n", ::capitalize("wayyyy downnn we gooooooooo!"));

    int    integers[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    double reals[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    integers + reals;

    return EXIT_SUCCESS;
}

template<class T, size_t length, class = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
[[nodiscard]] static constexpr long double sum(const T (&array)[length]) noexcept {
    long double sum {};
    for (const auto& e : array) sum += e;
    return sum;
}

constexpr int arr[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

static_assert(::sum(arr) == 45);
