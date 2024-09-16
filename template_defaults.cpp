#include <cstdlib>
#include <type_traits>

template<class _Type> constexpr auto declval() noexcept -> typename std::add_rvalue_reference_t<_Type>;

static_assert(std::is_same_v<decltype(.54564F), float>);

namespace utilities {
    template<class _TyL, class _TyR, class _TyS = _TyL> struct void_if_assignment_is_valid final {
            static constexpr bool value = false;
    };

    template<class _TyL, class _TyR> struct void_if_assignment_is_valid<_TyL, _TyR, decltype(::declval<_TyL>() = declval<_TyR>())> final {
            static constexpr bool value = true;
    };

    static_assert(void_if_assignment_is_valid<float&, double>::value);
} // namespace utilities

template<class _TyL, class _TyR, class _TySFINAE> struct is_assignable final {
        // we need _TySFINAE to become void when the evaluated assignment operation is valid, so the compiler will opt for the specialization
        // instead of the base template
        static constexpr bool value = false;
};

template<class _TyL, class _TyR> struct is_assignable<_TyL, _TyR, void /* when _TySFINAE = void */> final {
        using left_operand_type     = _TyL;
        using right_operand_type    = _TyR;
        static constexpr bool value = true;
};

// or we could choose to specialize on a bool instead of a type
template<class _TyL, class _TyR, bool is_valid> struct _is_assignable final {
        // we need _TySFINAE to become void when the evaluated assignment operation is valid, so the compiler will opt for the specialization
        // instead of the base template
        static constexpr bool value = false;
};

template<class _TyL, class _TyR> struct _is_assignable<_TyL, _TyR, true /* when is_valid = true */> final {
        using left_operand_type     = _TyL;
        using right_operand_type    = _TyR;
        static constexpr bool value = true;
};

auto wmain() -> int {
    int&& materialized_temporary { 11 };
    materialized_temporary = 12.05;

    return EXIT_SUCCESS;
}
