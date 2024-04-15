// g++ ctors.cpp -Wall -Wpedantic -Wextra -O3 -std=c++20

#include <cstdio>
#include <iostream>
#include <memory>

// pencil grades, prefixed with an underscore as many of them start with a number
enum class GRADE {
    UD /* undefined */,
    _9H,
    _8H,
    _7H,
    _6H,
    _5H,
    _4H,
    _3H,
    _2H,
    _H,
    _F,
    _HB,
    _B,
    _2B,
    _3B,
    _4B,
    _5B,
    _6B,
    _7B,
    _8B,
    _9B,
    _9xxB
};

std::wostream& operator<<(std::wostream& wostr, const GRADE& pencil_grade) {
    switch (pencil_grade) {
        case GRADE::UD    : wostr << L"UD"; break;
        case GRADE::_9H   : wostr << L"_9H"; break;
        case GRADE::_8H   : wostr << L"_8H"; break;
        case GRADE::_7H   : wostr << L"_7H"; break;
        case GRADE::_6H   : wostr << L"_6H"; break;
        case GRADE::_5H   : wostr << L"_5H"; break;
        case GRADE::_4H   : wostr << L"_4H"; break;
        case GRADE::_3H   : wostr << L"_3H"; break;
        case GRADE::_2H   : wostr << L"_2H"; break;
        case GRADE::_H    : wostr << L"_H"; break;
        case GRADE::_F    : wostr << L"_F"; break;
        case GRADE::_HB   : wostr << L"_HB"; break;
        case GRADE::_B    : wostr << L"_B"; break;
        case GRADE::_2B   : wostr << L"_2B"; break;
        case GRADE::_3B   : wostr << L"_3B"; break;
        case GRADE::_4B   : wostr << L"_4B"; break;
        case GRADE::_5B   : wostr << L"_5B"; break;
        case GRADE::_6B   : wostr << L"_6B"; break;
        case GRADE::_7B   : wostr << L"_7B"; break;
        case GRADE::_8B   : wostr << L"_8B"; break;
        case GRADE::_9B   : wostr << L"_9B"; break;
        case GRADE::_9xxB : wostr << L"_9xxB"; break;
        default           : break;
    }
    return wostr;
}

class pencil {
    private:
        float    price;
        wchar_t* make;
        GRADE    grade;

    public:
        // default ctor
        pencil() : price {}, make { new wchar_t[CHAR_MAX] }, grade { GRADE::_HB } { _putws(L"call to class pencil's default constructor"); }

        explicit pencil(const wchar_t* const wstr) : price {}, make { new wchar_t[CHAR_MAX] }, grade { GRADE::_HB } {
            wcsncpy_s(make, CHAR_MAX, wstr, _TRUNCATE);
            _putws(L"call to class pencil's pencil(const wchar_t* const wstr) constructor");
        }

        // copy ctor
        pencil(const pencil& other) : price { other.price }, make { new wchar_t[CHAR_MAX] }, grade { other.grade } {
            wcsncpy_s(make, CHAR_MAX, other.make, _TRUNCATE);
            _putws(L"call to class pencil's copy constructor");
        }

        // move ctor
        pencil(pencil&& other) noexcept : price { other.price }, make { other.make }, grade { other.grade } {
            other.price = 0.00F;
            other.make  = nullptr;
            other.grade = GRADE::UD;
            _putws(L"call to class pencil's move constructor");
        }

        pencil(const pencil&& other) = delete;

        // dtor
        ~pencil() noexcept {
            delete[] make;
            _putws(L"call to class pencil's destructor");
        }

        // copy assignment
        pencil& operator=(const pencil& other) {
            if (&other == this) return *this; // handling self assignment
            price = other.price;
            wcsncpy_s(make, CHAR_MAX, other.make, _TRUNCATE);
            grade = other.grade;
            _putws(L"call to class pencil's copy assignment operator");
            return *this;
        }

        // move assignment
        pencil& operator=(pencil&& other) noexcept {
            if (&other == this) return *this; // handling self assignment
            price       = other.price;
            make        = other.make;
            grade       = other.grade;

            other.price = 0.00F;
            other.make  = nullptr;
            other.grade = GRADE::UD;
            _putws(L"call to class pencil's move assignment operator");
            return *this;
        }

        friend std::wostream& operator<<(std::wostream& wostr, const pencil& object) {
            wostr << L"pencil { " << object.price << L", " << object.make << L", " << object.grade << L" }\n";
            return wostr;
        }
};

class pencilbox { };

int main() {
    auto atlas { pencil { L"Atlas" } };
    std::wcout << atlas;

    pencil camel;
    std::wcout << camel;

    camel = atlas;
    std::wcout << camel;
    std::wcout << atlas;

    pencil nataraj { L"Nataraj" };
    std::wcout << nataraj;

    camel = std::move(nataraj);
    std::wcout << camel;
    std::wcout << nataraj;

    return EXIT_SUCCESS;
}
