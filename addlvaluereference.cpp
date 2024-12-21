#include <type_traits>

namespace dwyer {
    template<typename _Ty, typename _TyPredicate> struct _add_lvref_helper final {
            // PRIMARY TEMPLATE, INTENDED AS A FALLBACK FOR VOID TYPES
            using type = _Ty;
    };

    template<typename _Ty> struct _add_lvref_helper<_Ty, std::remove_reference_t<_Ty&>> final {
            using type = _Ty&;
    };

    template<typename _Ty> using add_lvalue_reference_t = typename _add_lvref_helper<_Ty, _Ty>::type;
}

static_assert(std::is_same_v<void, dwyer::add_lvalue_reference_t<void>>);
static_assert(std::is_same_v<const void, dwyer::add_lvalue_reference_t<const void>>);
static_assert(std::is_same_v<volatile void, dwyer::add_lvalue_reference_t<volatile void>>);
static_assert(std::is_same_v<const volatile void, dwyer::add_lvalue_reference_t<const volatile void>>);
static_assert(std::is_same_v<long&, dwyer::add_lvalue_reference_t<long&>>);
static_assert(std::is_same_v<const float&, dwyer::add_lvalue_reference_t<const float>>);

static_assert(std::is_same_v<volatile double&, std::add_lvalue_reference_t<volatile double&&>>);
static_assert(std::is_same_v<volatile double&, dwyer::add_lvalue_reference_t<volatile double&&>>); // WHY???

static_assert(std::is_same_v<dwyer::_add_lvref_helper<volatile long&&, volatile long&&>::type, volatile long&>);
static_assert(std::is_same_v<dwyer::_add_lvref_helper<long&, long>::type, long>);

static_assert(std::is_same_v<const float&, dwyer::add_lvalue_reference_t<const float>>);
