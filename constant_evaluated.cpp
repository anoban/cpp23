// __cpp_lib_is_constant_evaluated

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>

template<typename T> [[nodiscard]] consteval typename std::enable_if<std::is_arithmetic_v<T>, T>::type materialize() noexcept {
    return T(M_2_SQRTPI);
}

auto wmain() -> int {
    const auto pi { M_PI };
    // what happens below is the compiler frst checks whether the subexpression on the true branch can be consteval evaluated
    // if yes, uses that value to initialize the variable statically
    // if not the variable is initialized with the false branch rvalue at RT

    // pi is a compile time constant, so value is statically initialized
    const auto value = std::is_constant_evaluated() ? pi : M_PI * 4;
    std::wcout << value << L"   " << M_PI * 4 << L'\n';

    volatile float pif { M_PI };
    // pif is not a compile time constant, hence the initialization is deferred to RT and is done using the false branch rvalue
    const auto     fvalue = std::is_constant_evaluated() ? pif : M_PI * 4;
    std::wcout << fvalue << L"   " << M_PI * 4 << L'\n';

    auto def = std::is_constant_evaluated() ? materialize<double>() : 0;
    // even though materialize() is consteval the variable cannot be initialized @ cmpile time because the variable definition is not a constant expression
    std::wcout << def << L"   " << 0 << L'\n';

    // when the definition is made a constant expression, materialize() is used for initialization
    const auto deff = std::is_constant_evaluated() ? materialize<double>() : 0;
    std::wcout << deff << L"   " << 0 << L'\n';

    return EXIT_SUCCESS;
}
