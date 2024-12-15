#include <sstring>

// all definitions are declarations but DECLARATIONS ARE NOT ALWAYS DEFINITIONS

static constexpr double __stdcall power(const float) noexcept; // DECLARATION

// this declaration gives nothing in terms of how the function behaves at runtime but gives enough insight to the compiler
// how it handles arguments and how it returns, hence this alone is enough for a compiler to check the validity of an invocation
// of this function

static double __cdecl square(const float x) noexcept { return x * x; } // this is a definition

// in C++, all object declarations are definitions unless it has an extern specifier AND no initializer
static int       COUNT;            // DEFINING DECLARATION
const float      pi { 3.1454897 }; // DEFINING DECLARATION
extern ::sstring anoban {};        // DEFINING DECLARATION
extern double*   _ptr;             // NON-DEFINING DECLARATION

// an object definition allocates storage for it while a declaration does not
::sstring name; // this does allocate storage for the ::sstring object named name

extern ::sstring other; // this does not allocate any storage for the ::sstring object

// it merely acknowledges its existence, a non-defining declaration says something exists but not here!

static auto wmain() -> int {
    //
    return EXIT_SUCCESS;
}
