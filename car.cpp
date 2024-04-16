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

std::wostream& operator<<(std::wostream& wostr, const Car& car) {
    wostr << car.brand << L' ' << car.model << L' ' << car.yom << L' ' << car.nseats << L' ' << car.hpower << L'\n';
    return wostr;
}

auto main() -> int {
    Car Camry; // default ctor gets called here
    std::wcout << Camry;

    Camry.brand  = L"Toyota";
    Camry.model  = L"Camry";
    Camry.yom    = 2012;
    Camry.nseats = 5;
    Camry.hpower = 2655.2;
    // well, that's one way to do it
    std::wcout << Camry;

    Car Allion {};  // value initialization, NOT A CALL TO DEFAULT CTOR
    std::wcout << Allion;

    return EXIT_SUCCESS;
}
