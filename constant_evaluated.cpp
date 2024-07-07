// __cpp_lib_is_constant_evaluated

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>

auto wmain() -> int {
    const auto pi { M_PI };
    const auto value = std::is_constant_evaluated() ? pi : M_PI * 4;
    std::wcout << value << L"   " << M_PI * 4 << L'\n';

    volatile float pif { M_PI };
    const auto     fvalue = std::is_constant_evaluated() ? pif : M_PI * 4;
    std::wcout << fvalue << L"   " << M_PI * 4 << L'\n';

    return EXIT_SUCCESS;
}
