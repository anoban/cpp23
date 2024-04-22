// clang .\closure.cpp -Wall -Wextra -std=c++20 -O3 -pedantic

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

// when a lambda is defined, the compiler creates a closure object
const auto printer = [](const std::wstring& wstring) noexcept -> void { ::_putws(wstring.c_str()); };

// compiler generated closure for the above lambda may look something like
struct __cpp20$$msvc15042024ucrt254_ThU202002LTtZ_printer$12__u35TiX_ {
        // since our lambda did not capture anything, we do not need any data members inside the functor
        void operator()(const std::wstring& wstring) const noexcept { ::wprintf_s(L"%s ", wstring.c_str()); }
};

// compiler generates a new closure object for each lambda, lambdas with identical signatures and function bodies will therefore be treated as different
// as they have different types e.g. struct _terribly_y5mangled_name_0 and struct __moreTerribly_mangled87Name_1 are distinct types
// consider the following lambdas, they differ only in their names
constexpr auto lambda_0 = [](int x, int y) constexpr noexcept -> int { return x + y; };
constexpr auto lambda_1 = [](int x, int y) constexpr noexcept -> int { return x + y; };

int main() {
    const std::wstring name { L"Anoban" };

    printer(name);
    __cpp20$$msvc15042024ucrt254_ThU202002LTtZ_printer$12__u35TiX_()(name); // ctor().operator()(name)

    constexpr auto closure { __cpp20$$msvc15042024ucrt254_ThU202002LTtZ_printer$12__u35TiX_ {} };
    closure(name);

    std::vector<std::wstring> tokens { L"This",  L"data",    L"sets",     L"consists",     L"of",  L"3",          L"different",    L"types",
                                       L"of",    L"irises'", L"(Setosa,", L"Versicolour,", L"and", L"Virginica)", L"petal",        L"and",
                                       L"sepal", L"length,", L"stored",   L"in",           L"a",   L"150x4",      L"numpy.ndarray" };

    std::for_each(tokens.cbegin(), tokens.cend(), printer);
    std::for_each(tokens.cbegin(), tokens.cend(), closure);

    static_assert(noexcept(lambda_0(8, 6456)));
    static_assert(noexcept(lambda_1(564, 1045)));

    // are the types of lambda_0 and lambda_1 same?
    static_assert(std::is_same_v<decltype(lambda_0), decltype(lambda_1)>, L"Will be different!");

    return EXIT_SUCCESS;
}
