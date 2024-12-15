#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>

class parent {
    protected:
        unsigned       length;
        unsigned char* buffer;

    public:
        parent() noexcept : length(::rand() % 1000), buffer(new (std::nothrow) unsigned char[length]) { }

        explicit parent(_In_ const unsigned& _size) noexcept : length(_size), buffer(new (std::nothrow) unsigned char[_size]) { }

        // how about a copy constructor
        parent(_In_ const parent& other) noexcept : length(other.length), buffer(new (std::nothrow) unsigned char[length]) { }

        // copy assignment operator
        parent& operator=(_In_ const parent& other) noexcept {
            ::_putws(L"" __FUNCSIG__);
            if (this == std::addressof(other)) return *this;
            length = other.length;
            delete[] buffer;
            buffer = new (std::nothrow) unsigned char[length];
            return *this;
        }

        ~parent() noexcept {
            ::_putws(L"" __FUNCSIG__);
            length = 0;
            delete[] buffer;
            buffer = nullptr;
        }

        template<typename _TyChar> friend std::basic_ostream<_TyChar>& operator<<(
            _Inout_ std::basic_ostream<_TyChar>& ostream, _In_ const parent& object
        ) noexcept(noexcept(ostream << object.buffer)) {
            ostream << std::hex << std::uppercase << object.buffer << static_cast<_TyChar>(' ') << std::dec << object.length
                    << static_cast<_TyChar>('\n');
            return ostream;
        }
};

// a child that does not have any members of its own
class child final : public parent {
    public:
        // WE DO NOT NEED TO EXPLICITLY INITIALIZE THE BASE CLASS HERE LIKE child() noexcept : parent() { }
        // NOR DO WE NEED TO DEFINE A TRIVIAL DEFAULT CONSTRUCTOR TO INITIALIZE THE BASE CLASS AS LONG AS WE DO NOT EXPLICITLY DEFINE ANY CONSTRUCTORS
        child() noexcept = default;

        // WE NO NOT NEED TO EXPLICITLY DEFINE A DESTRUCTOR EITHER
        // WHEN THE OBJECT GOES OUT OF SCOPE, THE COMPILER WILL INSTERT A CALL TO THE BASE CLASS'S DESTRUCTOR
        // DETERMINISTIC DESTRUCTION HUH :)

        // an explicit delegating constructor
        explicit child(_In_ const unsigned& _size) noexcept : parent(_size) { }

        // WITHOUT A USER DEFINED COPY CONSTRUCTOR, THE IMPLICITLY GENERATED COPY CONSTRUCTOR AUTOMATICALLY CALLS THE BASE CLASS'S COPY CONSTRUCTOR
        // HOW COOL IS THAT??
        // BUT THE BASE CLASS NEED TO HAVE A SEMANTICALLY RIGOROUS COPY CONSTRUCTOR, A DEFAULTED COPY CONSTRUCTOR OR AN IMPLICITLY GENERATED COPY
        // CONSTRUCTOR WILL LEAD TO BUGS WHEN THE CLASS HAS NON TRIVIAL DATA MEMBERS, E.G. SHALLOW COPIES OF HEAP ALLOCATED POINTERS MAY LEAD TO MEMORY LEAKS AND DOUBLE FREES

        // WITH A USER DEFINED COPY ASSIGNMENT OPERATOR, THE BASE CLASS'S COPY ASSIGNMENT OPERATOR WON'T BE AUTOMATICALLY INVOKED
        // child& operator=(_In_ const child& other) noexcept {
        //     // WE EXPLICTLY INVOKE THE BASE CLASS'S COPY ASSIGNMENT OPERATOR INSIDE THE DERIVED CLASS'S COPY ASSIGNMENT OPERATOR
        //     parent::operator=(other);
        //     return *this;
        // }

        // child& operator=(_In_ const child&) noexcept = default;
        ~child() noexcept { }
};

auto wmain() -> int {
    ::srand(::time(nullptr));

    child who { 54564 };
    std::wcout << std::setw(35) << L"who: " << who;
    who.~child();
    std::wcout << std::setw(35) << L"who.~child(): " << who;

    std::wcout << L'\n';

    child hello { 767 };
    std::wcout << std::setw(35) << L"hello: " << hello;
    auto copy { hello };
    // WITHOUT THE USER DEFINED COPY CONSTRUCTOR OF THE BASE CLASS, HERE THE COMPILER WILL USE AN IMPLICITLY GENERATED COPY
    // CONSTRUCTOR THAT WILL MAKE A SHALLOW COPY OF THE BASE CLASS, WHICH WILL LEAD TO A DOUBLE FREE SITUATION AT DESTRUCTION OF THE OBJECTS
    // AS BOTH WILL BE HOLDING A POINTER TO THE SAME HEAP ALLOCATED MEMEORY BLOCK
    std::wcout << std::setw(35) << L"copy: " << copy;

    std::wcout << L'\n';

    std::wcout << L"---------------------------------------------------------------------------------------\n";
    child default_constructed;
    std::wcout << std::setw(35) << L"default_constructed: " << default_constructed;
    default_constructed = hello; // WITHOUT A CUSTOM ROLLED COPY ASSIGNMENT OPERATOR OF THE BASE CLASS, THIS WILL LEAD TO THE LEAK
    // OF default_constructed'S BUFFER AND A DOUBLE FREE OF hello'S BUFFER
    // WHEN WE DON'T HAVE A USER DEFINED COPY ASSIGNMENT OPERATOR FOR THE BASE CLASS
    std::wcout << std::setw(35) << L"default_constructed = hello: " << default_constructed;
    std::wcout << std::setw(35) << L"hello: " << hello;
    std::wcout << L"---------------------------------------------------------------------------------------\n";

    {
        std::wcout << L"inside the block scope\n";
        const child jamie {};
        // EVEN WITH A USER DEFINED DESTRUCTOR FOR THE DERIVED CLASS, BASE CLASS'S DESTRUCTOR WILL BE INVOKED BY THE COMPILER
        // A CALL TO THE DESTRUCTOR OF THE DERIVED CLASS WILL INVOKE THE DESTRUCTOR OF THE BASE CLASS
    }
    std::wcout << L"outside the block scope\n\n";

    return EXIT_SUCCESS;
}
