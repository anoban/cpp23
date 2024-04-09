#include <type_traits>

// templated function without any explicit constraints
// implicit requirement = scalar_t operands must support the binary + operator
template<typename scalar_t> [[nodiscard]] static constexpr scalar_t sum(scalar_t x, scalar_t y) noexcept { return x + y; }

template<typename T> struct is_integral {
        static bool value;
};

// templated function with concepts
template<typename T> concept is_arithmetic = ;
