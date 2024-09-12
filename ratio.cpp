#include <iostream>
#include <ratio> // std::ratio is used to repreent fractions

// std::ratio operates strictly at compile time
// all custom std::ratio objects are templates
// we cannot use operators on them directly, utility templates provided in <ratio> must be used to perform
// arithmetics on the std::ratio templates

auto wmain() -> int {
    //
    using two_third = std::ratio<2, 3>;
    constexpr auto one { std::ratio_multiply<std::ratio<2, 3>, std::ratio<3, 2>> };

    return EXIT_SUCCESS;
}
