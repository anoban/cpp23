#include <type_traits>

struct fallback_type final { };

namespace dummy {

    template<typename _Ty, typename _TyPredicate> struct _add_lvref_helper final {
            using type = fallback_type; // PRIMARY TEMPLATE, INTENDED AS A FALLBACK FOR VOID TYPES
    };

    template<typename _Ty> struct _add_lvref_helper<_Ty, std::remove_reference_t<_Ty&>> final {
            using type = _Ty&;
    };

    template<typename _Ty> using add_lvalue_reference_t = typename _add_lvref_helper<_Ty, std::remove_reference_t<_Ty>>::type;
}

static_assert(std::is_same_v<::fallback_type, dummy::add_lvalue_reference_t<void>>);
static_assert(std::is_same_v<::fallback_type, dummy::add_lvalue_reference_t<const void>>);
static_assert(std::is_same_v<::fallback_type, dummy::add_lvalue_reference_t<volatile void>>);
static_assert(std::is_same_v<::fallback_type, dummy::add_lvalue_reference_t<const volatile void>>);

static_assert(std::is_same_v<long&, dummy::add_lvalue_reference_t<long&>>);
static_assert(std::is_same_v<const float&, dummy::add_lvalue_reference_t<const float>>);

static_assert(std::is_same_v<volatile double&, std::add_lvalue_reference_t<volatile double&&>>);
static_assert(std::is_same_v<volatile double&, dummy::add_lvalue_reference_t<volatile double&&>>);

static_assert(std::is_same_v<dummy::_add_lvref_helper<volatile long&&, volatile long&&>::type, fallback_type>);
static_assert(std::is_same_v<dummy::_add_lvref_helper<long&, long>::type, long&>);

static_assert(std::is_same_v<const float&, dummy::add_lvalue_reference_t<const float>>);

namespace test {

    template<class _Ty> struct bind final {
            static constexpr bool is_bound_to_primary = true;
    };

    template<class _Ty> struct bind<_Ty&> final {
            static constexpr bool is_bound_to_primary = false;
    };
}

static_assert(test::bind<double>::is_bound_to_primary);
static_assert(test::bind<double&&>::is_bound_to_primary); // in this case, if _Ty is deduced to be double&&
// here the specialization could be used too i.e. double& + & = double&, but the primary is a better match than the specialization
// in both cases, _Ty will be double&&
static_assert(!test::bind<double&>::is_bound_to_primary);
