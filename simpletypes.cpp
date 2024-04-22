// let's explore the thousand ways to initialize variables in C++
#include <array>
#include <memory>
#include <string>

struct vec3d {
        float x {}, y {}, z {};
};

static consteval vec3d Vec3DFactory() noexcept { return { 1.000, 1.000, 1.000 }; } // implicit construction of a vec3d object

int main() {
#pragma region INITIALIZATIONS

    int    a = 1;                              // plain old C style definition, i.e declaration and initialization in one line
    int    b { 2 };                            // list initialization
    double c                        = { 3.0 }; // copy list initialization

    auto                       heap = std::make_unique<float[]>(1000); // automatic type deduction
    constexpr auto             d { Vec3DFactory() };                   // automatic type deduction through a factory function
    std::basic_string<wchar_t> name { L"Anoban" };                     // ctor call
    constexpr vec3d            origin { 0.000 };                       // aggregate initialization
    std::array<char, 100>      e { 'A', 'r', 'r', 'a', 'y', ' ', 'i', 'n', 'i', 't', 'i', 'a', 'l', 'i', 'z', 'a', 't', 'i', 'o', 'n', 0 };
    // array initialization
#pragma end region

    return 0;
}
