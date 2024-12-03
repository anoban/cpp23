#include <iomanip>
#include <iostream>

static void dummy() noexcept {
    double        some {};                       // a regular variable
    auto&         refto_some { some };           // a reference to `some`
    auto*         pointerto_some { &some };      // pointer to `some`
    double* const constpointerto_some { &some }; // equivalent to refto_some because references are by definition const
    // thus analogous to constant pointers

    // assigning to a reference overwrites the referenced object, unlike pointers where the pointer will be modified
    short         height(10);
    auto&         refto_height { height };
    refto_height += 10; // height = 20 now
    auto* ptrto_height { &height };
    ptrto_height  += 10; // ptrto_height now points to a location 20 bytes right to where height is stored, height is still 10
    // to realize an equivalent effect with pointers,
    *ptrto_height += 10; // we need to explicitly dereference the pointer
}

// there are no const references because references are const by definition
// but there is a distinction between references to modifiable objects and const objects

static void example() noexcept {
    const auto marks { 98.4L };
    auto       age { 29UI8 };
    auto&      refto_constobj { marks };
    auto&      refto_modifiableobject { age };
}

// references are lvalues by definition, because when a reference is used, it yields an lvalue -> the object it refers to.

// the reason why C++ has references in addition to pointers to facilitate operator overloading
// consider the following enum
enum MONTHS { JANUARY, FEBRUARY, MARCH, APRIL, MAY, JUNE, JULY, AUGUST, SEPTEMBER, OCTOBER, NOVEMBER, DECEMBER };

static void months_increment() noexcept {
    // in plain C, one could do something like this
    for (MONTHS i = JANUARY; i < DECEMBER;
         ++i /* Error: this operation on an enumerated type requires an applicable user-defined operator function */)
        std::wcout << L"Yeehaw\n";
    // because in C, enums are just integers
    // C++ considers enums to be unique types, so to use the increment or decrement operators, they need to be overloaded
    const auto   ten { APRIL + 7 };
    const auto   five { FEBRUARY * 5 };
    const auto   six((DECEMBER + 1) / 2);
    // enums support all these arithmetic operations, then why not increment and decrement
    // THE DISTINCTION HERE IS THAT WE ARE OPERATING ON RVALUE TEMPORARIES YIELDED BY ENUMS
    // THE PROBLEM ARISES WHEN WE TRY TO TREAT ENUMS LIKE LVALUES!
    const MONTHS now { APRIL };     // okay
    static_assert(sizeof now == 4); // MONTHS uses int type
    now++;                          // Error: this operation on an enumerated type requires an applicable user-defined operator function
    // now look
    unsigned january { JANUARY }; // type is unsigned NOT MONTHS
    january++;                    // okay
    for (short i = JANUARY; i < DECEMBER; ++i) std::wcout << L"Yeehaw\n";
    // fine because we use a variable of type short and initialize it using an enum, ++ operator works on an short not on an enum
}

namespace incrementenums {
    enum class DAYS : unsigned char { MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY };

    // now we have another enum we'd like to support increment and decrement operators
    // increment and decrement perators in C and C++ behave like statements and exressions
    // they materialize in-place mutations and return the mutated value
    // e.g. x++ returns x and then increments x by 1
    // ++x increments x by 1 and then returns the new value

    [[nodiscard]] constexpr DAYS operator++(DAYS x) noexcept { return static_cast<DAYS>(static_cast<unsigned>(x) + 1U); }

    // the caveat with the above implementation is that it operates on values, it will of course return an incremented value
    // but the in-place mutation won't be materialized

    constexpr void               test() noexcept {
        auto today { DAYS::SATURDAY };
        ++today;        // today is still SATURDAY
        ++DAYS::SUNDAY; // incrementation on an rvalue! this shouldn't be possible
    }

    // using pointers is a possiblity but in C++ overloaded opeators cannot take pointers as arguments! so ....
    [[nodiscard]] constexpr DAYS operator++(DAYS* x) noexcept {
        *x = static_cast<DAYS>(static_cast<unsigned>(*x) + 1U);
        return static_cast<DAYS>(static_cast<unsigned>(*x) + 1U);
        // this implementation is theoretically sound but semantically wrong as C++ requires overloaded nonmember operator to have a parameter with class or enum types
        // but we use a pointer type here.
        // overloading ++ on pointer types obfuscates the usual semantics of pointer arithmetic.
        // BESIDES WITH POINTERS AS ARGUMENT TYPE. ++ NEEDS A POINTER TO BE THE RHS OPERAND
        // e.g. ++today; won't work! we'll need to use something like (&today)++;
    }

} // namespace incrementenums

static void rebind() noexcept {
    // references are like constant pointers, once bound to an object they cannot be rebound to another object
    auto  name { LR"(Anoban)" };
    auto& refwstr { name };
    auto  machine { LR"(Dell inspiron)" };
    refwstr = machine; // this doesn't modify the rference but modifies the pointer it refers to
    // since the object referred to by refwstr is a non-constant pointer it can be modified
    // AGAIN WE ARE NOT MUTATING THE REFERENCE ITSELF

    // IF A REFERENCE REFERS TO A CONSTANT POINTER, THIS WON'T BE POSSIBLE EVEN IF THE OBJECT IS NON-CONST
    long              age { 45 };      // a non-const variable
    auto&             refage { age };  // regular reference
    const auto&       crefage { age }; // const reference

    const auto* const cptrage { &age }; // a constant pointer to a constant long
    // we cannot modify age through cptrage, but age can be modified directly
    age      += 8; // Okay
    *cptrage /= 3; // Error: expression must be a modifiable lvalue

    refage   -= 8; // okay
    crefage  += 2; // Error: expression must be a modifiable lvalue

    // THE SAME GOES FOR OBJECTS THAT ARE CONST QUALIFIED THEMSELVES
}

auto main() -> int {
    srand(time(nullptr));
    long x { rand() }; // a varible of type long
    std::wcout << std::hex << std::uppercase << x << std::endl;

    auto  ptrx { &x }; // raw pointer to x
    auto& refx { x };  // reference to x

    std::wcout << ptrx << L' ' << refx << std::endl;
    // references are bogus objects, they do not exist at runtime like pointers

    // see what happens when we try to capture the address of the reference
    auto* addressx { &refx }; // this is supposed to be the address of the reference object

    std::wcout << addressx << std::endl;
    // this actually gives the address of the referenced object x

    if (ptrx == addressx) std::wcout << L"told ya!\n";
    return EXIT_SUCCESS;
}
