#include <iostream>

#define TOSTRING(expression)     (L#expression)       // stringisize
#define LINE(expression)         TOSTRING(expression) // helper to expand the __LINE__ macro

#define PRINT(integer_constexpr) printf_s(L"%d\n", ((##integer_constexpr)))

class T {
    public:
        T() noexcept { std::wcout << L"T()\n"; }
        T(int) throw() { std::wcout << __FUNCTIONW__ << L"(int)\n"; } // note the absence of the parameter name
};

int a = 87, b { 54 }; //globals

int main(void) {
    T(a);        // declaration of avariable a of type T, equivalent to T a;
    double(527); // function style cast
    T          v;
    const auto x { T(b) }; // construction using the myclass(int) constructor
    const auto z { T {} }; // construction using the myclass() constructor
    T          y();        // interpreted as a function declaration that takes no arguments and returns a T type
    // not a construction of T
    T          p(T(b)); // declaration of a function that takes an argument of type T and returns T
    // in the above line, T(b) can be interpreted as definition of a variable p of type T, constructed using the T(int) constructor
    // or as an argument of type T, with a redundant parenthesis around it, i.e T p(T b);

    // if the intention is to declare an object of type T, use an explicit static cast
    T P(static_cast<T>(b));
    T Q { b }; // or use the braced initializer

    std::wcout << LINE(__LINE__) << std::endl;
    PRINT(89);

    return EXIT_SUCCESS;
}
