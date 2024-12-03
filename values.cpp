#include <array>
#include <cmath>
#include <limits>
#include <numbers>

// practiced following Ben Sak's Cppcon 2019 talk on C++ value categories.
// value categories are not language features, they are heuristics usd by the compiler to understand storage type information

// when they were first introduced in C, they were relatively simple and straightforward concepts!
// C has just two value categories - lvalues and rvalues
// lvalue stands for left values in an assignment or a definition
// rvalues stands for right values in an assignment or a definition

// and C++ came along, and unsuprisingly made it more complicated, as always!
// with modern C++, value categories became exponentially more convoluted :) classic C++ stuff.

struct coordinate {
        double               x {};
        double               y {};
        std::array<float, 4> z {};
};

consteval coordinate GetMaxCoord() noexcept {
    return {
        .x = std::numeric_limits<long double>::max(), .y = FLT_MAX, .z = { std::numeric_limits<float>::min() }
    }; // since we have an explicit return type, specifying it after the return keyword like return coordinate { };
    // becomes redundant!
}

int main() {
    long  x { 12345 }; // a variable definition
    // long x declares a variable of type long and gives it the identifier x
    // at this point the variable x has secured a storage of sizeof(long)
    // { 12345 } writes (stores) the value 12345 to the storage of variable x
    // x is an lvalue, an identifier with addressable storage (not registers)
    // 12345 is an rvalue, a literal with no addressable storage

    // NOTE THE TERM "ADDRESSABLE STORAGE"
    // BY ADDRESSABLE STORAGE WE MEAN THAT A VARIABLE'S ADDRESS COULD BE CAPTURED BY A POINTER
    // A VALUE TEMPORARILY STORED IN A CPU REGISTER DOES HAVE A STORAGE, BUT WE CANNOT CAPTURE ITS ADDRESS USING C/C++
    // WITH x64 WE COULD SIMPLY USE THE REGISTER, BUT FROM C/C++ PERSPECIVE, WE CANNOT!

    float f32pi; // declaration
    // variable declaration secures the asked type of storage for the given variable
    f32pi = std::numbers::pi_v<float>; // assignment
    // assignment writes the provided object/value to the lvalue's storage
    // f32pi is an lvalue and 3.14159265358979 is an rvalue

    unsigned squares[10] { 0, 1, 4, 9, 16, 25, 36, 49, 64, 81 };
    // squares is an lvalue (an array of lvalues, if you will)
    squares[0] = 100; // squares[0] gives us an lvalue, NOTE THAT THE OFFSET INSIDE THE SUBSCRIPT OPERATOR IN ITSELF IS AN RVALUE LITERAL
    // AND WE UPDATE THE VALUE AT squares[0], as it points to a modifiable and addressable storage

    squares[3LLU * 3] = 49; // here the subexpression inside the subscript operator is an rvalue that evaluates to 9LLU
    // and squares[3LLU * 3] yields a modofiable lvalue!

    // NOT ALL IDENTIFIERS WITH A STORAGE ARE MODIFIABLE! IDENTIFIERS QUALIFIED WITH CONST WILL BE STORED IN READ-ONLY PAGES
    // AND CANNOT BE OVERWRITTEN! THEY ARE CONSIDERED LVALUES BUT THEY CANNOT BE ASSIGNED TO OR OVERWRITTEN
    const auto f64pi { std::numbers::pi_v<double> };
    // here f64pi is an lvalue with storage, but it is a const qualified object in read-only memory!
    auto       ptrf64pi { &f64pi }; // that's the captured address of the identifier f64pi
    f64pi = 1.234567890;            // compile time error - the left expression must be a modofiable lvalue
                                    // THE KEY PHRASE HERE IS "MODIFIABLE LVALUE"
                                    // FOR AN IDENTIFIER TO BE ASSIGNED TO, IT MUST BE A MODIFIABLE LVALUE

    // consider another definition with a slightly more convoluted rvalue expression
    auto origin {
        coordinate { 0.000, 0.000, 0.000 }
    };
    auto ptr_origin { &origin };

    squares[static_cast<unsigned>(pow(2.0L, 4.0L)) - 10] = static_cast<unsigned>(fmax(ptr_origin->x, ptr_origin->y));
    // let's dissect this
    // squares[static_cast<unsigned>(pow(2.0L, 4.0L)) - 10] is a modifiable lvalue, granted the subexpression inside the subscript operator
    // doesn't lead to a buffer overrun!
    // static_cast<unsigned>(pow(2.0L, 4.0L)) - 10 is an rvalue that doesn't have a addressable storage.
    // sure the programme will have to evaluate this expression at runtime and ultimately come up with an integer literal
    // but it doesn't have to give that literal an addressable storage!

    // static_cast<unsigned>(fmax(ptr_origin->x, ptr_origin->y)) is an rvalue.
    // a temporary without a addressable storage.

    // WHY DOES THIS DISTINCTION EXIST??
    // VALUE CATEGORIES ALLOW COMPILERS TO GENERATE MORE EFFICIENT MACHINE CODE GRANTED THAT THE LANGUAGE SPECS DO NOT EXPECT RVALUES TO
    // HAVE ADDRESSABLE STORAGE.
    // IF RVALUES WERE TO HAVE ADDRESSABLE STORAGE, THE MACHINE CODE MUST INCLUDE INSTRUCTIONS TO COMPUTE THE TEMPORARIES AND MOVE THEM
    // TO A STACK LOCATION (TO MAKE THE TEMPORARY ADDRESSABLE) AND USE IT IN THE SUBSEQUENT COMPUTATIONS.
    // THIS WILL BE TERRIBLY INEFFICIENT AS THERE IS GRATUITOUS COPIES IN AND OUT OF REGISTERS AND THE STACK.

    // SINCE THIS IS NOT THE CASE, THE COMPILER COULD CHOOSE TO KEEP THE TEMPORARIES IN REGISTERS AND USE THEM AS IT SEES FIT
    // THIS AVOIDS GRATUITOUS COPIES TO AND FROM STACK AND REGISTERS
    // AND FOR LITERALS, THESE VALUES COULD SIMPLY BE HARDCODED INTO THE MACHINE INSTRUCTIONS AS WE DO NOT REQUIRE ITS STORAGE BE ADDRESSABLE!
    // E.G mov n, 45    ; store 45 at n's memory location
    // IF LITERLAS (RVALUES) HAD TO OCCUPY STORAGE, THE COMPILER WOULDN'T BE ABLE TO DO THIS!
    // THE CODE WOULD HAVE LOOKED SOMETHING LIKE,
    // .DATA
    // n                                 DWORD ?
    // _@@tmp@@rvalue_08497cpp_rt2023xx_ DWORD 42   ; giving the rvalue a storage
    // .CODE
    // .....
    // mov n, _@@tmp@@rvalue_08497cpp_rt2023xx_

    const int dummy                                      = 12;
    1 = dummy; // ERROR: expression must be a modifiable lvalue, expression is not assignable
    // we know the above line is invalid, BUT WHY?
    // 1 and d have the same types (int)
    // but 1 is an rvalue that doesn't have a storage. We cannot write to a memory location that is not addressable!

    // this is true for builtin types, but the situation is a little more nuanced and complicated for class types
    // i.e classes, structs and unions

    // SO, WHAT TYPES OF LITERALS ARE RVALUES?
    // INTEGERS, FLOATS, CHARACTERS
    // STRING LITERLAS ARE LVALUES!! EVEN THOUGH THEIR MEMORY IS READ ONLY, THEY ARE LVALUES AS WE NEED TO BE ABLE TO SUBSCRIPT INTO
    // STRING LITERALS AND THIS WILL NOT BE POSSIBLE WITHOUT THE STRING HAVING A BASE ADDRESS SO THE COMPILER CAN GET US THE
    // REQUESTED CHARACTER FROM (BASE_ADDR + OFFSET)

    const auto*    str { "STRING LITERAL" };
    constexpr auto path { R"(C:\Users\Jamie\Documents\Books)" }; // raw string literal
    auto           J { "SAMUEL JACKSON"[9] };
    // above, we indexed into an anonymous string literal. BECAUSE IT HAD AN ADDRESSABLE YET READ-ONLY STORAGE
    // i.e the literal had a base address, so the 10th element could be captured by offsetting 9 bytes right of the base address
    "NATALIE"[4] = 'L'; // ERROR: expression must be a modifiable lvalue, read-only variable is not assignable

    // OPERANDS ON THE RIGHT HAND SIDE OF AN ASSIGNMENT OR DEFINITION OR INITIALIZATION DOESN'T NEED TO BE A RVALUE
    // E.G.
    auto       eps { 1.000E-4L };
    const auto threshold { eps }; // the operand on the right hand side is an lvalue.
    // C++ semantics say that when an lvalue is used as the right jhand operand in an assignment/definition/initialization, an
    // lvalue to rvalue conversion takes place.

    // operands of an arithmetic operator can be either an  rvalue or an lvalue.
    eps += std::numbers::inv_pi_v<float>; // here the right operand of the + operator is an rvalue
    eps -= threshold; // here the right operand of the - operator is an lvalue, which ensues an LVALUE TO RVALUE CONVERSION

    // the temporary that results from evaluating the RHS of the assignment operation is also an rvalue
    // i.e eps -= threshold; expands as eps = eps - threshold;
    // evaluation of the subexpression eps - threshold results in an rvalue temporary.

    // pointer indirection operator unary *, yields an lvalue
    unsigned    number { 100 };
    auto* const numptr { &number };
    *numptr *= 3; // is an lvalue - has an addressable storage
    // dereferecing a pointer yields an lvalue (can be a read only lvalue BUT STILL AN LVALUE)
    constexpr auto cost { 1'340'599.99L };
    auto*          costptr { &cost }; // pointer to a constant varible, lvalue but unmodifiable
    *costptr /= 36.6435F;             // Error: expression must be a modifiable lvalue, read-only variable is not assignable

    int* thevoid { nullptr }; // pointer to an integer
    int  ten { 10 };
    // dereferencing thevoid still gives a valid lvalue, from a language semantics perspective.
    // but will result in undefined behaviour.

    thevoid = &ten; // valid, because the pointer thevoid is an lvalue - has an addressable and modifiable storage

    thevoid = nullptr;
    *thevoid /* supposed to yield a modifiable lvalue, but dereferences a nullptr */ = 212;
    // will probably raise an access violation exception

    // when operands become large, like class types, the penalty of increased binary size may outweigh the benefits of inlining the values in binary
    // the compiler may opt to give the temporaries a storage, but will not expose it's address to users or programmers
    // but programmers are expected to program as if this doesn't happen

    auto           origin_0 { GetMaxCoord() };
    constexpr auto FLOAT32_MAX { GetMaxCoord().y }; // GetMaxCoord() is expected to return a temporary rvalue
    // but GetMaxCoord().y requests a member of the returned struct
    // assemblers find struct members using a base address and an offset to the requested member variable.
    // in order to have a base address, we need to store the type in memory i.e the stack or the heap.
    // so in this case, an lvalue temporary will be materialized in stack, before offsetting to find out the member variable y.
    // BECAUSE CLASS TYPE RVALUES DO OCCUPY DATA STORAGE
    constexpr auto FLOATMIN { GetMaxCoord().z[2] }; // same story here

    // ENUMERATIONS ARE RVALUES, LIKE INTEGER LITERALS
    enum class PREDICATES : unsigned char { NO, MAYBE, YES };
    unsigned char* whereNO { &(PREDICATES::NO) }; // Error: cannot take the address of an rvalue of type 'PREDICATES'
    PREDICATES::YES *= 17;                        // Error: expression is not assignable
    // Error: in C++ enums are special types, not regular ints as in C, so to use a compound *= operator, it must be oberloaded to support the PREDICATES type

    // use of value categories in reference types
    int    integer { 1000 };
    int&   ref { integer }; // unlike pointers, references function like an alias for the referred variable
    // unlike pointers, references can only refer to valid existing objects (no NULL references)
    // one cannot even declare a reference without providing a valid object as right operand.
    float& floatref;            // Error:  reference variable "floatref" requires an initializer
    float& nullref { nullptr }; // Error: a value of type "std::nullptr_t" cannot be used to initialize an entity of type "float"

    auto   where_ref { &ref }; // note the type, it is an int*, NOT int**
    // because &ref actually yields the address of the object that the reference points to, not the address of the reference itself.
    return 0;
}
