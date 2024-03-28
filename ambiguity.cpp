#include <iostream>

class myClass {
    public:
        myClass() { std::wcout << L"constructing myClass\n"; }
        myClass(int) { std::wcout << L"constructing myClass(int)\n"; } // note the absence of the parameter name
};

int a = 87, b { 54 }; //globals

int main(void) {
    myClass(a);                  // declaration of avariable a of type myClass
    myClass    v;
    const auto x { myClass(b) }; // construction
    const auto z { myClass {} };
    myClass    y();              // interpreted as a function declaration that takes no arguments and returns a myClass type
    // not a construction of myClass
    myClass    p(myClass(b));

    return EXIT_SUCCESS;
}
