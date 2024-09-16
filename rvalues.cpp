#include <cstdlib>
#include <iostream>
#include <numbers>
#include <type_traits>

static inline void func(_Inout_ double& val) noexcept { val *= 2.000; }

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type> class pair final {
    private:
        T first;
        T second;
        template<typename, typename> friend class pair; // makes all instantiations of template pair<> friends

    public:
        constexpr pair() noexcept : first(), second() { }

        constexpr explicit pair(const T& _val) noexcept : first(_val), second(_val) { }

        constexpr explicit pair(const T& _val0, const T& _val1) noexcept : first(_val0), second(_val1) { }

        constexpr ~pair() = default;

        // copy ctor, move ctor, copy assignment and move assignment operators cannot be templates!
        constexpr pair(const pair& other) noexcept : first(other.first), second(other.second) { }

        constexpr pair(pair&& other) noexcept : first(other.first), second(other.second) { }

        constexpr pair& operator=(const pair& other) noexcept {
            if (&other == this) return *this;
            first  = other.first;
            second = other.second;
            return *this;
        }

        constexpr pair& operator=(pair&& other) noexcept {
            if (&other == this) return *this;
            first  = other.first;
            second = other.second;
            return *this;
        }

        // universal ctor
        // without the friend declaration, pair<T> cannot access the private members of pair<U> because these two are two completely
        // different classes in their post-instantiation state!
        template<typename U>
        constexpr pair(const ::pair<U>& other) noexcept : first(static_cast<T>(other.first)), second(static_cast<T>(other.second)) { }

        template<typename char_t> friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const pair& object) {
            // using function style casts
            ostream << char_t('{') << char_t(' ') << object.first << char_t(',') << char_t(' ') << object.second << char_t(' ')
                    << char_t('}');
            return ostream;
        }

        // consider operator overloading
        template<typename U> [[nodiscard]] constexpr pair<long double> operator+(_In_ const pair<U>& other) const& noexcept {
            // this operator+ is for const pair<T>& types, so operator+() will work with both lvalues and rvalues
            // an operator+ with signature template<typename U> [[nodiscard]] constexpr pair<long double> operator+(_In_ const pair<U>& other) & noexcept
            // will only be usable with a non const pair<T>& lvalue type
            return pair<long double> { first + other.first, second + other.second };
        }

        template<typename U> [[nodiscard]] constexpr pair<long double> operator-(_In_ const pair<U>& other) & noexcept {
            // will work only when the left operand is of type ::pair<T>& (a non-const lvalue reference)
            return pair<long double> { first - other.first, second - other.second };
        }

        template<typename U> [[nodiscard]] constexpr pair<long double> operator/(_In_ const pair<U>& other) && noexcept {
            // will work only when the left operand is of type ::pair<T>&& (a non-const rvalue reference)
            return pair<long double> { static_cast<long double>(first) / other.first, static_cast<long double>(second) / other.second };
        }
};

auto wmain() -> int {
    constexpr ::pair<float> fpair { std::numbers::pi_v<float> };
    std::wcout << fpair << L'\n';

    constexpr auto spair { ::pair<short> {} };
    std::wcout << spair << L'\n';

    constexpr auto dpair {
        ::pair<double> { 12.086, 6543.0974 }
    };
    std::wcout << dpair << L'\n';

    [[maybe_unused]] constexpr auto zee { dpair };
    ::pair<float>                   q {};
    q = fpair;
    std::wcout << q << L'\n';

    constexpr ::pair<long> lpair { fpair }; // calls the templated copy ctor
    std::wcout << lpair << L'\n';

    q = q;

    [[maybe_unused]] constexpr auto sum { ::pair<float> { std::numbers::pi } + ::pair<unsigned char> {} }; // both operands are rvalues
    [[maybe_unused]] constexpr auto _sum { fpair + ::pair<unsigned char> { 3 } }; // left operand const lvalue, right operand rvalue
    [[maybe_unused]] constexpr auto _sum_ { fpair + fpair };                      // both operands are const lvalues

    constexpr auto error { fpair - ::pair<float> {} }; // won't work because the left operand is const but operator-
    // takes an non-const ::pair<T>& as the left operand

    constexpr auto error_too { ::pair<double> { std::numbers::pi } -
                               ::pair<float> {} }; // will not work because the left operand of operator-
    // has to be an non const lvalue pair<T>& but the argument we provided is an rvalue

    constexpr auto wontcompile { fpair / ::pair<double> { 1.000 } };
    // left operand is const pair<T>& (const lvalue reference) but operator/ expects a non-const rvalue reference
    // non const method cannot operate on const lvalues

    const auto wontcompile_either { q / ::pair<double> { 1.000 } }; // left operand is non const but an lvalue
    // operator/ expects an rvalue for left operand, hence the error

    [[maybe_unused]] constexpr auto okay { ::pair<float> { std::numbers::pi_v<float> } / fpair };
    ::pair<double>                  egamma { std::numbers::egamma }; // non const lvalue
    [[maybe_unused]] const auto     okay_too { ::pair<float> { egamma - fpair } };

    const auto x { ::func(2.000) };

    return EXIT_SUCCESS;
}

// rvalues of builtin types (literals) do not occupy storage, but once they are bound to T&& or const T& they will qualify for a storage space
// there are two types of rvalues :: prvalues and xvalues
// prvalues do not occupy data storage but xvalues do!

// temporary materializations convert a prvalue into an xvalue
static void __declspec(noinline) __stdcall function() noexcept {
    const ::pair<double> invsqrtpi { std::numbers::inv_sqrtpi };

    ::pair<float>&&     temp { invsqrtpi }; // this is a materialization of a converting constructed temporary
    // FIRST WE CONSTRUCT A TEMPORARY OF TYPE ::pair<float> FROM AN LVALUE OF TYPE const ::pair<double>
    // THEN WE BIND THAT TEMPORARY TO THE IDENTIFIER `temp`
    const ::pair<char>& _temp { invsqrtpi }; // same
    // temp and _temp are xvalues that will be destroyed at the end of this function scope
    // BUT FOR CLASS TYPES BOTH TEMPORARIES AND LVALUES HAVE A MEMORY LOCATION

    /// since ::pair<T> is a class type, their rvalues were already occupying storage before we captured them with T&& or const T&
    // CONSIDER PRIMITIVE TYPES
    const long double& value { 0.276334587145 }; // here the literal 0.276334587145 does not have a memory address
    // but the temporary materialized from it does!
    // value is an xvalue while the literal is an rvalue
}
