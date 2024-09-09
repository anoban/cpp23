#include <concepts>
#include <cstdlib>
#include <iostream>

// references and pointers do not allow type punned bindings!

[[maybe_unused]] static float eps { 2.71828182845905 };

template<class T> requires std::is_arithmetic_v<T> class wrapper final {
    private:
        T _value;

    public:
        constexpr wrapper() noexcept : _value {} { }

        constexpr explicit wrapper(const T& _init) noexcept : _value { _init } { }

        wrapper(const wrapper&)            = delete; // no copy ctor
        wrapper& operator=(const wrapper&) = delete; // no copy assignment operator
        wrapper(wrapper&&)                 = delete; // no move ctor
        wrapper& operator=(wrapper&&)      = delete; // no move assignment operator

        constexpr ~wrapper() noexcept { _value = 0; }

        // conversion operator
        template<class _Ty> requires std::is_arithmetic_v<_Ty> operator _Ty() noexcept { return static_cast<_Ty>(_value); }
};

auto wmain() -> int {
    // try binding a eps to a double&
    // double&            invalid = eps;
    // but eps can be bound to a const type& as the compiler can create a type casted temporarya and bind it to the reference
    const long double& valid { eps };   // okay, but this reference does not point to the object eps!!!
    long long&&        valid_too = eps; // cannot use a braced initializer here because of narrowing :(

    std::wcout << std::hex << std::uppercase;
    std::wcout << std::addressof(eps) << L'\n' << std::addressof(valid) << L'\n' << std::addressof(valid_too) << L'\n';

    std::wcout << L"\n\n";

    auto                      pi { wrapper { 3.14159265358979L } };
    const wrapper<float>&     okay { pi };
    wrapper<unsigned short>&& okay_too { pi };
    std::wcout << std::addressof(pi) << L'\n' << std::addressof(okay) << L'\n' << std::addressof(okay_too) << L'\n';

    return EXIT_SUCCESS;
}
