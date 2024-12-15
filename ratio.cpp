#include <iomanip>
#include <iostream>
#include <ratio> // std::ratio is used to represent fractions

// std::ratio are templates that operate strictly at compile time
// all custom std::ratio objects are templates
// we cannot use operators on them directly, utility templates provided in <ratio> must be used to perform arithmetics on the std::ratio templates

template<class char_type, int64_t numerator, int64_t denominator> std::basic_ostream<char_type>& operator<<(
    std::basic_ostream<char_type>& ostream, const std::ratio<numerator, denominator>& ratio
) noexcept(noexcept(ostream << ratio.num)) {
    ostream << ratio.num << char_type('/') << ratio.den << char_type('\n');
    return ostream;
}

template<class char_type, int64_t numerator, int64_t denominator> std::basic_ostream<char_type>& operator<<=(
    std::basic_ostream<char_type>& ostream, const std::ratio<numerator, denominator>& ratio
) noexcept(noexcept(ostream << ratio.num)) {
    ostream << static_cast<double>(ratio.num) / ratio.den << char_type('\n');
    return ostream;
}

inline namespace { // handrolled std::ratio alternative

    template<__int64 _rnumer, __int64 _rdenom> struct ratio final {
            static constexpr __int64 num { _rnumer };
            static constexpr __int64 den { _rdenom };

            // NOLINTNEXTLINE(google-explicit-constructor) - enable implicit conversion to real types
            template<typename T> requires std::floating_point<T> [[nodiscard]] constexpr operator T() const noexcept {
                return static_cast<T>(num) / static_cast<T>(den);
            }

            template<typename char_t> friend std::basic_ostream<char_t>& operator<<(
                _In_ const ratio& fraction, _Inout_ std::basic_ostream<char_t>& ostr
            ) noexcept(noexcept(ostr << num)) {
                ostr << fraction.num << char_t('/') << fraction.den << char_t('\n');
                return ostr;
            }

            template<__int64 __rnum, __int64 __rden> constexpr decltype(auto) operator+(_In_ const ratio<__rnum, __rden>&) const noexcept {
                return ratio<_rnumer + __rnum, _rdenom + __rden> {};
            }
    };

    template<__int64 _rnumer> struct ratio<_rnumer, 0> final {
            // a ratio template with 0 as denominator is prone to divide by zero errors
            // hence, leaving this specialization empty (could have also used a statc_assert inside the main template :p)
    };
} // namespace

auto wmain() -> int {
    std::wcout << std::fixed << std::setprecision(5);

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
