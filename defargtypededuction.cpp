#include <string>
#include <type_traits>

template<typename T> struct phone {
        typename std::enable_if<std::is_unsigned_v<T>, T>::type _yom { 2023 };
        decltype(_yom)                                          _price { 12'000 };
        std::wstring                                            _make;
        std::wstring                                            _model;
        std::wstring                                            _country;
};

template<typename aggregate_type, typename T>
static constexpr T func(
    const aggregate_type& _agg       = phone<T> {},
    const T&              _soldunits = 561 // template type deduction cannot use default args to deduce types
) noexcept(std::is_nothrow_constructible_v<typename phone<T>>) {
    _soldunits;
    _agg;
}

template<typename scalar_t = unsigned, typename aggregate_type = phone<scalar_t>>
static constexpr scalar_t _func(
    const aggregate_type& _agg       = phone<scalar_t> {},
    const scalar_t&       _soldunits = 561u // template type deduction cannot use default args to deduce types
) noexcept(std::is_nothrow_constructible_v<typename phone<scalar_t>>) {
    return _soldunits * _agg._price;
}

int main() {
    //
    constexpr auto result  = func();
    constexpr auto _result = _func();
    return EXIT_SUCCESS;
}
