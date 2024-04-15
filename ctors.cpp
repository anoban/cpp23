// g++ ctors.cpp -Wall -Wpedantic -Wextra -O3 -std=c++20

#include <cstdio>
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

class pencil {
    private:
        float    price;
        wchar_t* make;
        GRADE    grade;

    public:
        // default ctor
        pencil() : price {}, make { new wchar_t[CHAR_MAX] }, grade { GRADE::_HB } {
            _putws(L"call to class pencil's default constructor\n");
        }

        pencil(const wchar_t* const wstr) : price {}, make { new wchar_t[CHAR_MAX] }, grade { GRADE::_HB } {
            wcsncpy_s(make, CHAR_MAX, wstr, _TRUNCATE);
            _putws(L"call to class pencil's pencil(const wchar_t* const wstr) constructor\n");
        }

        // copy ctor
        pencil(const pencil& other) : price { other.price }, make { new wchar_t[CHAR_MAX] }, grade { other.grade } {
            wcsncpy_s(make, CHAR_MAX, other.make, _TRUNCATE);
            _putws(L"call to class pencil's copy constructor\n");
        }

        // move ctor
        pencil(pencil&& other) : price { other.price }, make { std::move(other.make) }, grade { other.grade } {
            other.price = 0.00F;
            other.grade = GRADE::UD;
            _putws(L"call to class pencil's move constructor\n");
        }

        pencil(const pencil&& other) = delete;

        // dtor
        ~pencil() noexcept {
            delete[] make;
            _putws(L"call to class pencil's destructor\n");
        }

        // copy assignment
        pencil& operator=(const pencil& other) {
            if (&other == this) return *this; // handling self assignment
            price = other.price;
            wcsncpy_s(make, CHAR_MAX, other.make, _TRUNCATE);
            grade = other.grade;
            _putws(L"call to class pencil's copy assignment operator\n");
            return *this;
        }

        // move assignment
        pencil& operator=(pencil&& other) noexcept {
            if (&other == this) return *this; // handling self assignment
            price       = other.price;
            make        = std::move(other.make);
            grade       = other.grade;
            other.price = 0.00F;
            other.grade = GRADE::UD;
            _putws(L"call to class pencil's move assignment operator\n");
            return *this;
        }
};

class pencilbox { };

int main() {
    auto   atlas { pencil { L"Atlas" } };
    pencil camel;
    camel = atlas;

    pencil nataraj { L"Nataraj" };
    camel = std::move(nataraj);

    return EXIT_SUCCESS;
}
