#include <iostream>
#include <numbers>

class base {
    private:
        long private_long;

    protected:
        long protected_long;

    public:
        long public_long;

        constexpr base() noexcept : private_long { 10 }, protected_long { 100 }, public_long { 100 } { }
};

class pubderived : public base {
        // a derived class that will have access to base class's all protected and public members
        // but cannot access base class's private members

    private:
        float priv_float;

    public:
        // ctor
        constexpr pubderived() noexcept : base {}, priv_float { std::numbers::pi_v<float> } { }
        // upon construction of the derived class type, call the constructor of the base class.

        constexpr double get() const noexcept { return private_long + protected_long + public_long + priv_float; }
};

class privderived : base {
    private:
        double priv_double;

    public:
        // ctor
        constexpr privderived() noexcept : base {}, priv_double { std::numbers::pi_v<double> } { }
        // upon construction of the derived class type, call the constructor of the base class.

        constexpr double get() const noexcept { return private_long + protected_long + public_long + priv_double; }
};
