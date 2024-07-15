#include <cstdlib>
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

        constexpr double& unwrap() noexcept;

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

constexpr double& object::unwrap() noexcept { return _value; }

constexpr const double& object::unwrap() const noexcept { return _value; }

class mobject {
    public:
        mutable double _value; // the only distinction between class object and mobject is that the member _value is mutable in mobject
        // and _value is a public member

        constexpr mobject() noexcept;

        constexpr explicit mobject(const double& _init) noexcept;

        constexpr mobject(const mobject& other) noexcept;

        constexpr mobject(mobject&& other) noexcept;

        constexpr mobject& operator=(const mobject& other) noexcept;

        constexpr mobject& operator=(mobject&& other) noexcept;

        constexpr ~mobject() noexcept;

        constexpr double& unwrap() const noexcept;

        constexpr void update(const double& value) const noexcept;
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

constexpr void mobject::update(const double& value) const noexcept { _value = value; }

auto wmain() -> int {
    constexpr object pi { std::numbers::pi_v<double> };
    // pi.unwrap() += pi.unwrap();
    // constexpr mobject meps { std::numbers::e_v<double> };

    const mobject meps { std::numbers::e_v<double> };
    meps.unwrap() += meps.unwrap();

    meps._value   *= 2.00;
    meps.update(7856.65);

    constexpr struct {
            int          one;
            mutable long two;
    } anonymous { -12, 36768 };

    anonymous.one++; // ERROR
    anonymous.two++; // OKAY because of the mutable qualifier

    return EXIT_SUCCESS;
}
