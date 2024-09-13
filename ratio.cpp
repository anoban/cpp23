#include <iomanip>
#include <iostream>
#include <ratio> // std::ratio is used to represent fractions

// std::ratio are templates that operate strictly at compile time
// all custom std::ratio objects are templates
// we cannot use operators on them directly, utility templates provided in <ratio> must be used to perform arithmetics on the std::ratio templates

template<class char_type, int64_t numerator, int64_t denominator>
std::basic_ostream<char_type>& operator<<(std::basic_ostream<char_type>& ostream, const std::ratio<numerator, denominator>& ratio)
    noexcept(noexcept(ostream << ratio.num)) {
    ostream << ratio.num << char_type('/') << ratio.den << char_type('\n');
    return ostream;
}

template<class char_type, int64_t numerator, int64_t denominator>
std::basic_ostream<char_type>& operator<<=(std::basic_ostream<char_type>& ostream, const std::ratio<numerator, denominator>& ratio)
    noexcept(noexcept(ostream << ratio.num)) {
    ostream << static_cast<double>(ratio.num) / ratio.den << char_type('\n');
    return ostream;
}

template<template<int64_t, int64_t> class _ratio> requires requires {
    _ratio<1, 1>::num;
    _ratio<1, 1>::den;
} struct to_wstr final {
        static constexpr const wchar_t* operator()() const noexcept { static wchar_t ascii[10] {}; }
};

auto wmain() -> int {
    std::wcout << std::setprecision(5);

    using two_third = std::ratio<2, 3>;
    using one_third = std::ratio<1, 3>;

    constexpr auto one { std::ratio_multiply<std::ratio<2, 3>, std::ratio<3, 2>> {} };
    std::wcout << one;

    std::wcout << std::ratio_divide<two_third, one_third> {}; // 2/1
    std::wcout << std::ratio_add<two_third, one_third> {};    // 1/1

    std::wcout <<= std::ratio_divide<two_third, one_third> {}; // 2.000
    std::wcout <<= std::ratio_add<two_third, one_third> {};    // 1.000

    return EXIT_SUCCESS;
}
