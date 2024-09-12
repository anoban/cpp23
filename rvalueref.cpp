#include <concepts>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <ranges>

// references and pointers do not allow type punned bindings!

[[maybe_unused]] static float eps { 2.71828182845905 }; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

template<class T> requires std::is_arithmetic_v<T> class wrapper final {
    public:
        using value_type = T;

    private:
        T _value;
        template<class _Ty> requires std::is_arithmetic_v<_Ty>
        friend class wrapper; // to help different typed instantiations of this class template cross access private members

    public:
        constexpr wrapper() noexcept : _value {} { }

        constexpr explicit wrapper(const T& _init) noexcept : _value { _init } { ::_putws(L"" __FUNCSIG__); }

        wrapper(const wrapper&)            = delete; // no copy ctor
        wrapper& operator=(const wrapper&) = delete; // no copy assignment operator
        wrapper(wrapper&&)                 = delete; // no move ctor
        wrapper& operator=(wrapper&&)      = delete; // no move assignment operator

        constexpr ~wrapper() noexcept { _value = 0; }

        // conversion operator
        template<class _Ty> requires std::is_arithmetic_v<_Ty> operator _Ty() const noexcept {
            ::_putws(L"template<class _Ty> requires std::is_arithmetic_v<_Ty> operator _Ty() noexcept");
            return static_cast<_Ty>(_value);
        }

        // when a templated copy ctor is available, the compiler uses it instead of a conversion operator
        // because technically a use of conversion operator as defined here requires invocation of the copy ctor (as defined here) down the line
        // with the acquired wrapped value type
        template<class _Ty> constexpr wrapper(const wrapper<_Ty>& other) noexcept : _value { static_cast<T>(other._value) } {
            ::_putws(L"template<class _Ty> requires std::is_arithmetic_v<_Ty> constexpr wrapper(const wrapper<_Ty>& other) noexcept");
        }

        template<> constexpr wrapper<unsigned char>(const wrapper<unsigned char>&) noexcept = delete;
        // for unsigned chars the compiler will err saying call to a deleted ctor
        // instead of falling back to using the conversion operator & non-templated copy ctor
};

template<> template<> constexpr ::wrapper<double>::wrapper(const wrapper<unsigned>&) noexcept = delete;

template<std::floating_point T> static constexpr T e_v                                        = static_cast<T>(2.71828182845905L);

[[nodiscard]] static inline constexpr double pi() noexcept { return 3.14159265358979; }

auto wmain() -> int {
    // try binding a eps to a double&
    // double&            invalid = eps;
    // but eps can be bound to a const type& as the compiler can create a type casted temporarya and bind it to the reference
    const long double& valid { eps };   // okay, but this reference does not point to the object eps!!!
    long long&&        valid_too = eps; // cannot use a braced initializer here because of narrowing :(

    std::wcout << std::hex << std::uppercase;
    std::wcout << std::addressof(eps) << L'\n' << std::addressof(valid) << L'\n' << std::addressof(valid_too) << L'\n';

    std::wcout << L'\n';

    auto                      pi { wrapper { 3.14159265358979L } };
    const wrapper<float>&     okay { pi };
    wrapper<unsigned short>&& okay_too { pi };

    std::wcout << std::addressof(pi) << L'\n' << std::addressof(okay) << L'\n' << std::addressof(okay_too) << L'\n';

    const unsigned short& truncation = std::numbers::pi_v<double>;

    const float*       ptr           = &std::numbers::pi_v<float>; // works
    // const double*      _ptr          = &2.71828182845905;          //  error: cannot take the address of an rvalue of type 'double'
    const long double* __ptr         = &::e_v<long double>; // works, seems like we can take the address of constexpr'd variable templates

    // const double* _pi                = &::pi(); // WOW

    // invocation of conversion operator and non-template copy ctor
    const ::wrapper<unsigned char> ucpi { 3 };
    const ::wrapper<unsigned>      ui32pi { std::numeric_limits<unsigned>::max() };
    const ::wrapper<double>        what { ucpi };
    const ::wrapper<double>        should_be_okay { double(ucpi) };
    const ::wrapper<double>        should_not_be_okay { ui32pi }; // call to a deleted ctor

    // binding an rvalue reference T&& to a prvalue of type T triggers a temporary materialization!
    [[maybe_unused]] unsigned&& rvref { 87245 };

    // we could also have a const rvalue reference, but it is not something that's particularly useful
    const std::wstring&& crvref { L"What can I do with this?" }; // this might as well have been a const std::wstring object

    return EXIT_SUCCESS;
}

static inline constexpr double&& power(const float& base, const unsigned& exp) noexcept {
    if (!exp) return 1.0;
    if (exp == 1) return base;
    double temp { base };
    for (const auto& _ : std::ranges::views::iota(1U, exp)) temp *= base;
    return static_cast<double&&>(temp); // without the cast, an lvalue cannot be returned as an rvalue reference
    // casting is stupid because "read of temporary whose lifetime has ended"
}

static_assert(::power(std::numbers::egamma, 0) == 1.0000);
