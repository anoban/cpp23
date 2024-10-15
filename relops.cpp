#include <concepts>
#include <cstdlib>
#include <type_traits>

// NOLINTBEGIN(modernize-use-constraints,readability-redundant-inline-specifier)

namespace derived_comparisons { // expects that the passed types suppport == and < operators

    template<class _TyL, class _TyR>
    static inline constexpr typename std::enable_if_t<std::equality_comparable_with<_TyL, _TyR>, bool> operator!=(
        const _TyL& left, const _TyR& right
    ) noexcept {
        return !(left == right);
    }

    template<class _TyL, class _TyR> static inline constexpr bool operator<=(const _TyL& left, const _TyR& right) noexcept
        requires requires(const _TyL& left, const _TyR& right) {
            left == right;
            left < right;
        } {
        return left == right || left < right;
    }

    template<class _TyL, class _TyR> requires std::equality_comparable_with<_TyL, _TyR>
    static inline constexpr bool operator>(const _TyL& left, const _TyR& right) noexcept {
        return !(left == right || left < right); // return !dreived_comparisons::operator<=(left, right);
    }

    template<class _TyL, class _TyR> static inline constexpr bool operator>=(const _TyL& left, const _TyR& right) noexcept {
        return !(left < right);
    }

} // namespace derived_comparisons

// NOLINTEND(modernize-use-constraints,readability-redundant-inline-specifier)

int wmain() {
    using namespace derived_comparisons;
    return EXIT_SUCCESS;
}
