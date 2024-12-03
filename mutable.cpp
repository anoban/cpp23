// g++ mutable.cpp -Wall -O3 -std=c++20 -municode -Wextra -Wpedantic -o mutable.exe
// clang .\mutable.cpp -Xclang -fdump-record-layouts -Wall -Wextra -std=c++20 -c -Xclang -fdump-vtable-layouts -O3

#include <cstdlib>
#include <iostream>
#include <numbers>
#include <utility>

class object {
    private:
        double _value;

    public:
        constexpr object() noexcept;

        constexpr explicit object(const double& _init) noexcept;

        constexpr object(const object& other) noexcept;

        constexpr object(object&& other) noexcept;

        constexpr object& operator=(const object& other) noexcept;

        constexpr object& operator=(object&& other) noexcept;

        constexpr ~object() noexcept;

        constexpr double&       unwrap() noexcept;

        constexpr const double& unwrap() const noexcept;
};

constexpr object::object() noexcept : _value {} { }

constexpr object::object(const double& _init) noexcept : _value { _init } { }

constexpr object::object(const object& other) noexcept : _value { other._value } { }

constexpr object::object(object&& other) noexcept : _value { other._value } { other._value = 0.00; }

constexpr object& object::operator=(const object& other) noexcept {
    if (std::addressof(other) == this) return *this;
    _value = other._value;
    return *this;
}

constexpr object& object::operator=(object&& other) noexcept {
    if (std::addressof(other) == this) return *this;
    _value       = other._value;
    other._value = 0.00;
    return *this;
}

constexpr object::~object() noexcept { _value = 0.00; }

constexpr double&       object::unwrap() noexcept { return _value; }

constexpr const double& object::unwrap() const noexcept { return _value; }

class mobject {
        mutable double _value; // the only distinction between class object and mobject is that the member _value is mutable in mobject

    public:
        constexpr mobject() noexcept;

        constexpr explicit mobject(const double& _init) noexcept;

        constexpr mobject(const mobject& other) noexcept;

        constexpr mobject(mobject&& other) noexcept;

        constexpr mobject& operator=(const mobject& other) noexcept;

        constexpr mobject& operator=(mobject&& other) noexcept;

        constexpr ~mobject() noexcept;

        constexpr double& unwrap() const noexcept;

        constexpr void    update(const double& value) const noexcept;
};

constexpr mobject::mobject() noexcept : _value {} { }

constexpr mobject::mobject(const double& _init) noexcept : _value { _init } { }

constexpr mobject::mobject(const mobject& other) noexcept : _value { other._value } { }

constexpr mobject::mobject(mobject&& other) noexcept : _value { other._value } { other._value = 0.00; }

constexpr mobject& mobject::operator=(const mobject& other) noexcept {
    if (std::addressof(other) == this) return *this;
    _value = other._value;
    return *this;
}

constexpr mobject& mobject::operator=(mobject&& other) noexcept {
    if (std::addressof(other) == this) return *this;
    _value       = other._value;
    other._value = 0.00;
    return *this;
}

constexpr mobject::~mobject() noexcept { _value = 0.00; }

constexpr double& mobject::unwrap() const noexcept { return _value; }

constexpr void    mobject::update(const double& value) const noexcept { _value = value; }

template<typename T, typename char_t>
requires std::is_same_v<char, char_t> ||
         std::is_same_v<wchar_t, char_t>
         static inline std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const T& object)
             requires requires { object.unwrap(); } {
    ostream << object.unwrap() << char_t('\n');
    return ostream;
}

auto wmain() -> int {
    constexpr object pi { std::numbers::pi_v<double> };
    // pi.unwrap() += pi.unwrap();

    std::wcout << pi;

    // constexpr mobject meps { std::numbers::e_v<double> };
    // assignment to mutable member '_value' is not allowed in a constant expression, in call to 'meps.~mobject()':
    const mobject meps { std::numbers::e_v<double> }; // cannot make this a constexpr, welp :(
    std::wcout << meps;
    meps.unwrap() += meps.unwrap();
    std::wcout << meps;

    meps.unwrap() *= 2.00;
    std::wcout << meps;
    meps.update(std::numbers::pi_v<double>);
    std::wcout << meps;

    constexpr struct {
            int          one;
            mutable long two;
    } anonymous { -12, 36768 };

    // anonymous.one++; // cannot assign to variable 'anonymous' with const-qualified type 'const struct (unnamed struct)
    anonymous.two++; // OKAY because of the mutable qualifier

    constexpr auto explicit_ctor_call = object(std::numbers::ln10);
    std::wcout << explicit_ctor_call;

    // in C++, we cannot call constructors directly because constructors do not have names, so they are never found during name lookup;
    return EXIT_SUCCESS;
}
