#include <functional>
#include <iostream>

// callables passed to std::bind1st and std::bind2nd must be a `AdaptableBinaryFunction`, a plain binary function does not work

static unsigned sum(char x, unsigned long y) noexcept { return x + y; }

int main() {
    const auto summ = std::bind(sum, 'd', std::placeholders::_1);
    std::wcout << summ(100L) << L'\n';

    const auto subtract34 = std::bind1st(std::minus<int>(), 34);
    std::wcout << subtract34(34) << L'\n'; // 0
    std::wcout << subtract34(68) << L'\n'; // -34
    std::wcout << subtract34(0) << L'\n';  // 34

    // const auto addQ = std::bind2nd(sum, 'Q');    // using raw function pointers does not work with std::bindnth family of functions
    // they expect a callable of type AdaptableBinaryFunction
    // std::ptr_fun can convert a regular function into AdaptableBinaryFunction!
    // BUT WHY THE FUCK THOUGH?
    const auto addQ = std::bind2nd(std::ptr_fun(&sum), 'Q');
    std::wcout << addQ(19LU) << L'\n'; // 100?

    return EXIT_SUCCESS;
}
