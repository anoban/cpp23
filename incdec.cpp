#include <concepts>
#include <iostream>

struct integral {
        unsigned v;

        integral() = delete; // delete the default ctor

        explicit constexpr integral(unsigned value) noexcept : v { value } { }

        constexpr integral& operator++() noexcept { // prefix++ operator
            v++;
            return *this; // prefixed ++ operator first increments the value and then returns the incremented value
        }

        constexpr integral& operator--() noexcept { // prefix--
            v--;
            return *this;
        }

        // since posfix ++ and -- operators retun the object prior to mutating it, we cannot return a reference to self
        // because at the return statement, self would have already been mutated
        // we cannot return from a function and then mutate a member
        constexpr integral operator++(int) noexcept { // postfix++ operator
            v++;
            return integral { v - 1 }; // postfixed ++ operator first returns the value and then increments the value in-place
        }

        constexpr integral operator--(int) noexcept { // postfix--
            v--;
            return integral { v + 1 };
        }

        friend std::wostream& operator<<(std::wostream& wostr, integral& integ) {
            wostr << integ.v << L'\n';
            return wostr;
        }

        friend std::wostream& operator<<(std::wostream& wostr, const integral&& integ) {
            wostr << integ.v << L'\n';
            return wostr;
        }
};

int main() {
    auto twelve { integral { 12 } };
    std::wcout << ++twelve;
    std::wcout << twelve; // calls friend std::wostream& operator<<(std::wostream& wostr, integral& integ) with an lvalue reference

    std::wcout << integral { 44 };
    // calls friend std::wostream& operator<<(std::wostream& wostr, const integral&& integ) with a xvalue reference

    std::wcout << --twelve;
    std::wcout << --twelve;
    std::wcout << twelve; // 11

    int ten { 10 };
    std::wcout << ten << L' ' << ten++ << L' ' << ten << L'\n'; // 10 10 11
    // ten is 11 now
    std::wcout << ten << L' ' << ++ten << L' ' << ten << L'\n'; // 11 12 12

    ++ten = 0;
    std::wcout << ten << L'\n'; // if prefix ++ or -- operators return a reference this must print 0
    // and it does print 0!

    std::wcout << twelve++; // 11
    std::wcout << twelve++; // 12
    std::wcout << twelve;   // 13
    std::wcout << twelve--; // 13
    std::wcout << twelve--; // 12
    std::wcout << twelve;   // 11

    return 0;
}
