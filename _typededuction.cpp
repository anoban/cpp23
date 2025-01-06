#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <type_traits>

template<typename _Ty> static void values_only([[maybe_unused]] _Ty _argument) noexcept {
    ::wprintf_s(L"Is _Ty a reference? %4s\n", std::is_reference_v<decltype(_argument)> ? L"Yes" : L"No");
}

template<typename _Ty> static void references_only([[maybe_unused]] _Ty& _argument) noexcept {
    ::wprintf_s(L"Is _Ty a reference? %4s\n", std::is_reference_v<decltype(_argument)> ? L"Yes" : L"No");
}

int wmain() {
    constexpr auto value { std::numbers::pi_v<float> };
    const auto&    lvalue_reference { value };
    const auto&&   rvalue_reference { 2.718281828459045 };

    ::values_only(value);
    ::values_only(lvalue_reference);
    ::values_only(rvalue_reference);

    ::references_only(value);
    ::references_only(lvalue_reference);
    ::references_only(rvalue_reference);

    return EXIT_SUCCESS;
}
