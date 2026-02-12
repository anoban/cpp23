#include <iostream>

// a non-static member function can be declared virtual in the base class
// this will deem member functions in all the derived classes with the same name virtual
// deriving classes doesn't necessarily have to declare their member function as virtual but it usually is a good practice to do so.

class base {
    protected:
        double b { 12.00 }; // private members of the base class are not accessible to the derived classes

    public:
        // this is our virtual member function for the base class
        constexpr virtual double get() const noexcept { return b; }

        // virtual functions in the derived classes need to have the same name and argument types
};

// when a derived class privately inherits (the default unless explicitly stated public inheritance) from the base class
// base's protected members are deemed private in the derived class
class middle : public base {
    protected:
        double m { 12.00 };

    public:
        // overriden virtual function
        constexpr virtual double get() const noexcept override { return m + b; }
};

class top final : public middle {
    protected:
        double t { 12.00 };

    public:
        constexpr virtual double get() const noexcept override { return b + m + t; }
};

int main() {
    constexpr base   bObject {};
    constexpr middle mObject {};
    constexpr top    tObject {};

    std::wcout << L"base.get() = " << bObject.get() << L" middle.get() = " << mObject.get() << L" top.get() = " << tObject.get()
               << std::endl;

    ::puts("Let's locate the vtable pointers!");
    // these pointers are supposed to be stored in an array of function pointers
    printf("base::get() %X\n", &base::get);
    printf("middle::get() %X, middle::base::get() %X, top::get() %X\n", &middle::get, &middle::base::get);
    ::printf(
        L"top::get() %X, top::middle::get() %X, top::middle::base::get() %X\n", &top::get, &top::middle::get, &top::middle::base::get
    );

    return EXIT_SUCCESS;
}
