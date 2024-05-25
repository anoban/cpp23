// in C++, there are two flavours of class data members - static and non-static
// and three flavours of class function members - static, non-static and virtual

#include <cstdlib>
#include <iostream>

class object {
    private:
        float value; // non static data member

    public:
        static short sh; // static data member - same for all instances of class object in a programme

        constexpr object() : value(3.14159265358979) { }

        // non-static member function
        constexpr float get() const noexcept { return value; }

        // non-static member function
        constexpr float operator()() const noexcept { return value; }

        // static member function, static member functions are not associated with class instances
        // but with the class definition
        static constexpr float pi() noexcept { return 3.14159265358979; }

        // a virtual dtor
        virtual ~object() = default;
};

int wmain() {
    object         obj;
    constexpr auto pi { ::object::pi() };
    // but static member functions can also be accessed through class instances
    constexpr auto pi2 { obj.pi() };
    std::wcout << pi << L' ' << pi2 << L'\n';
    std::wcout << object::sh;

    return EXIT_SUCCESS;
}
