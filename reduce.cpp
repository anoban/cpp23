#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

template<typename _TyIterator, typename _TyBinaryOperator, typename _TyInit> // NOLINTNEXTLINE(modernize-use-constraints)
static constexpr typename std::enable_if<
    std::is_arithmetic<typename _TyIterator::value_type>::value &&
        std::is_assignable<typename std::add_lvalue_reference<_TyInit>::type, typename _TyIterator::value_type>::value,
    _TyInit>::type
reduce(_In_ _TyIterator _begin, _In_ const _TyIterator& _end, _In_ const _TyInit& _init, _In_ _TyBinaryOperator _binop) noexcept {
    _TyInit result { _init };
    while (_begin != _end) {
        result = _binop.operator()(result, *_begin);
        ++_begin;
    }
    return result;
}

int main() {
    std::mt19937_64     rengine { std::random_device {}() };
    std::vector<double> numbers(3'000);
    std::generate(numbers.begin(), numbers.end(), rengine);

    const auto sum { ::reduce(numbers.cbegin(), numbers.cend(), 0.00L, std::plus<long double> {}) };
    std::cout << std::setw(15) << "::reduce " << std::setprecision(20) << sum << '\n';
    std::cout << std::setw(15) << "std::reduce " << std::setprecision(20) << std::reduce(numbers.cbegin(), numbers.cend()) << '\n';

    return EXIT_SUCCESS;
}
