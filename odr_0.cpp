#include <numbers>

extern unsigned           external_linkage;

const double              pie { std::numbers::pi }; // in C++ const variables have internal linkage (i.e behave like static variables)

__declspec(noinline) void function() noexcept { external_linkage *= 2; }
