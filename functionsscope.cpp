#include <cstdlib>
#include <iostream>

static constexpr double RMAX { RAND_MAX };
double                  (*fnptr)(const double&) noexcept {};

int wmain() {
    ::srand(::time(nullptr));

    {
        // scope of the lambda
        auto square = [](const double& _value) constexpr noexcept -> double { return _value * _value; };
        fnptr       = square;
    }

    double random {};

    std::wcout << L"square of " << (random = ::rand() / RMAX) << L" is " << ::fnptr(random) << L'\n';

    return EXIT_SUCCESS;
}
