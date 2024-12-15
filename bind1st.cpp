#include <functional>
#include <iostream>

// a predicate that returns true if the passed double is less than 100.00
const auto predicate_lt       = std::bind2nd(std::less<double>(), 100.00);

// second predicate thate returns true if the passed double is greater than or equals to 73.00
const auto predicate_ge       = std::bind2nd(std::greater_equal<double>(), 73.00);

// a negation operation - (x < 73.00) i.e !(x >= 73.00)
const auto negate_ge          = std::bind(std::logical_not<bool>(), predicate_ge, std::placeholders::_1);

// composed predicate - !(x >= 73.00) && (x < 100.00)
const auto composed_predicate = std::bind(std::logical_and<bool>(), negate_ge, predicate_lt, std::placeholders::_1);

int main() {
    double x { 50.00 };
    std::wcout << std::boolalpha;
    for (int i = 0; i < 100; i += 15)
        std::wcout << "!(" << i + x << L" >= 73.00) && (" << i + x << L" < 100.00) - " << negate_ge(x + i) << L'\n';
    return EXIT_SUCCESS;
}
