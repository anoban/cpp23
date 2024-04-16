#include <functional>
#include <iostream>

// lambdas can be wrapped in std::function<> objects
// but this approach is rarely necessary and almost always incurs a performance penalty
// std::functions is a rather complex wrapper that can handle all sorts of functions, not only lambdas so it has many bells and whistles to it
// internals of std::function involves type punning and dynamic memory allocation if needed

class functor {
    public:
        constexpr int operator()(int x, int y) const noexcept { return x * y; }
};

// a regular function
static constexpr double sumd(int x, int y) noexcept { return static_cast<double>(x) + y; }

// an equivalent lambda
constexpr auto          lambda = [](int x, int y) constexpr noexcept -> double { return static_cast<double>(x) + y; };

// type signature of both functions is essentially => constexpr double (int, int) noexcept;

int                     main() {
    constexpr auto                  x { 11 }, y { -1 };

    std::function<double(int, int)> wrapper = sumd;
    // std::function wrapper requires the type signature of the function as template arguments
    // qualifiers like constexpr, consteval and noexcept should not be included in the template arguments

    std::wcout << L"sum(" << x << L", " << y << L") = " << sumd(x, y) << L'\n';
    std::wcout << L"sum(" << x << L", " << y << L") = " << wrapper(x, y) << L'\n';
    std::wcout << L"sum(" << x << L", " << y << L") = " << lambda(x, y) << L'\n';

    static_assert(std::is_same<decltype(0.000), double>::value);
    // decltype gives the type of the passed object NOT THE RETURN TYPE

    std::wcout << L"size of lambda is " << sizeof(lambda) << " bytes!\n";

    std::function<double(int, int)> lambda_wrapper = lambda;
    std::wcout << L"size of the same lambda wrapped in a std::function class object is " << sizeof(lambda_wrapper) << " bytes!\n";

    std::wcout << L"size of bare-bones functor is " << sizeof(functor) << " bytes!\n";

    return EXIT_SUCCESS;
}
