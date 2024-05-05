#include <type_traits>

template<bool, typename T> struct predicate_if { }; // declaration

template<typename T> struct predicate_if<true, T> {
        using type = T;
        static constexpr bool value { true };
};

// alias template
template<bool pred, typename T> using predicate_if_t          = ::predicate_if<pred, T>::type;

// variable template
template<bool pred, typename T> constexpr bool predicate_if_v = ::predicate_if<pred, T>::value;

namespace numbers {
    template<class T>
    constexpr typename ::predicate_if<std::is_floating_point<T>::value, T>::type pi_v = static_cast<T>(3.141592653589793L);
    template<typename T, typename ::predicate_if_t<std::is_floating_point_v<T>, T> = T(0.0)>
    constexpr T avogadro_v = static_cast<T>(6.02214076E23L);
} // namespace numbers

int main() {
    constexpr auto                                      v { ::predicate_if<true, float>::value };
    constexpr typename ::predicate_if_t<2 == 2, double> dbl {};
    constexpr auto                                      boolean { ::predicate_if_v<(23 > 11), const char> };
    constexpr auto                                      area { ::numbers::pi_v<double> * 12 * 12 };
    constexpr auto                                      avo { ::numbers::avogadro_v<float> };
    return 0;
}
