// in C++, there are two flavours of class data members - static and non-static
// and three flavours of class function members - static, non-static and virtual

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>

#pragma pack(push, 1)
class cylinder {
    private:
        float _radius; // non static data members
        float _height;

    public:
        static short _statmember; // static data member - same for all instances of class object in a programme

        constexpr cylinder() noexcept : _radius(), _height() { }

        constexpr explicit cylinder(const float& _r, const float& _h) noexcept : _radius(_r), _height(_h) { }

        constexpr cylinder(const cylinder& _other) noexcept : _radius(_other._radius), _height(_other._height) { }

        constexpr cylinder(cylinder&& _other) noexcept : _radius(_other._radius), _height(_other._height) {
            _other._radius = _other._height = 0.0000;
        }

        constexpr cylinder& operator=(const cylinder& _other) noexcept {
            if (this == &_other) return *this;
            _radius = _other._radius;
            _height = _other._height;
            return *this;
        }

        constexpr cylinder& operator=(cylinder&& _other) noexcept {
            if (this == &_other) return *this;
            _radius        = _other._radius;
            _height        = _other._height;
            _other._radius = _other._height = 0.0000;
            return *this;
        }

        // a virtual dtor
        virtual ~cylinder() = default;

        // non-static member function
        constexpr float area() const noexcept { return M_PI * _radius * _radius + _radius * _height; }

        // non-static member function
        constexpr float volume() const noexcept { return M_PI * _radius * _radius * _height; }

        // static member function, static member functions are not associated with class instances
        // but with the class definition
        static constexpr float pi() noexcept { return M_PI; }

        template<typename char_t>
        friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& _ostr, const cylinder& _object) {
            _ostr << _object._radius << char_t(' ') << _object._height << char_t('\n');
            return _ostr;
        }
};
#pragma pack(pop)

short cylinder::_statmember = std::numeric_limits<short>::max();

cylinder static_initialization;
// since this initialization is outside main(), who initializes this object at runtime and when?
// C++ standard gurantees that global objects will be initialized before their first use in the programme
// this means somewhere inside the main() the compiler will squeeze in the ctor call to this object to make it valid before its first use!
// where this happens is often at the discretion of the compilers

int wmain() {
    cylinder       object;
    constexpr auto pi { ::cylinder::pi() };
    // but static member functions can also be accessed through class instances
    constexpr auto pivalue { object.pi() };

    std::wcout << pi << L' ' << pivalue << L'\n';
    std::wcout << std::hex << std::uppercase << cylinder::_statmember << L'\n';

    std::wcout << L"&object = " << &object << L" std::addressof(object) = " << std::addressof(object) << L'\n';
    std::wcout << L"sizeof(object) = " << sizeof(object) << L" std::addressof(cylinder::_statmember) = "
               << std::addressof(cylinder::_statmember) << L'\n';

    return EXIT_SUCCESS;
}
