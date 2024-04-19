// g++ car.cpp -Wall -Wextra -Wpedantic -O3 -std=c++11

#include <iostream>
#include <string>

struct Car {
        // all are non-static member variables
        std::wstring  brand; // make of the car
        std::wstring  model;
        unsigned      yom; // year of manufacture
        unsigned char nseats;
        float         hpower;
};

struct Bus {
        // all are non-static member variables
        std::wstring  brand; // make of the car
        std::wstring  model;
        unsigned      yom; // year of manufacture
        unsigned char nseats;
        float         hpower;

        Bus() : brand { L"Mercedes-Benz" }, model { L"OH 1830 RF" }, yom { 2024 }, nseats { 44 }, hpower { 7200.0 } { }
};

std::wostream& operator<<(std::wostream& wostr, const Car& car) {
    wostr << car.brand << L' ' << car.model << L' ' << car.yom << L' ' << car.nseats << L' ' << car.hpower << L'\n';
    return wostr;
}

std::wostream& operator<<(std::wostream& wostr, const Bus& bus) {
    wostr << bus.brand << L' ' << bus.model << L' ' << bus.yom << L' ' << bus.nseats << L' ' << bus.hpower << L'\n';
    return wostr;
}

auto main() -> int {
    Car Camry; // default ctor (compiler provided) gets called here
    std::wcout << Camry;

    Camry.brand  = L"Toyota";
    Camry.model  = L"Camry";
    Camry.yom    = 2012;
    Camry.nseats = 5;
    Camry.hpower = 2655.2;
    // well, that's one way to do it
    std::wcout << Camry;

    Car Allion {}; // value initialization, NOT A CALL TO DEFAULT CTOR
    // IF CLASS Car HAD A USER PROVIDED DEFAULT CTOR, VALUE INITIALIZATION WOULD HAVE CALLED THAT CTOR
    std::wcout << Allion;

    Bus b, bb {};
    std::wcout << b;
    std::wcout << bb;

    return EXIT_SUCCESS;
}
