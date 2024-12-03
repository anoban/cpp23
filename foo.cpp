#include <cstdlib>
#include <type_traits>

constexpr auto yes { std::is_integral_v<decltype(7645)> };
constexpr auto no { std::is_integral_v<decltype(764.5)> };

template<typename T, bool> struct foo;

template<typename T> struct foo<T, true> final {
    public:
        constexpr foo() noexcept : _rsrc {} { }

        constexpr explicit foo(const T& _value) noexcept : _rsrc { _value } { }

    private:
        T _rsrc;
};

template<typename T, bool = std::is_integral_v<T>> struct bar;

template<typename T> struct bar<T, true> final {
    public:
        bar() = delete;

        constexpr explicit bar(const T& _value) noexcept : _rsrc { _value } { }

    private:
        T _rsrc;
};

template<int I, int J> struct A { };

template<int I> struct A<I + 5, I * 2> { }; // error, I is not deducible

template<int I, int J, int K> struct B { };

template<int I> struct B<I, I * 2, 2> { }; // OK: first parameter is deducible

auto wmain() -> int {
    constexpr auto x { foo { 125 } };
    constexpr auto y { bar<unsigned> { 1252 } };
    return EXIT_SUCCESS;
}
