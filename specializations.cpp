#include <string>

namespace type_traits {

    // THESE ARE PARTIAL SPECIALIZATIONS NOT EXPLICIT SPECIALIZATIONS!
    // instead of specializing for concrete types we specialize on variants of the template parameter _Ty

    template<class _Ty> struct is_reference final {
            static constexpr bool value = false;
    };

    template<class _Ty> struct is_reference<_Ty&> final { // the specializing type _Ty& is a dependent of the template type parameter _Ty
            static constexpr bool value = true;
    };

    template<class _Ty> struct is_reference<_Ty&&> final { // the specializing type _Ty&& is a dependent of the template type parameter _Ty
            static constexpr bool value = true;
    };

    template<class _Ty> using add_lvalue_reference_t = _Ty&;

    template<class _Ty> using add_rvalue_reference_t = _Ty&&;

} // namespace type_traits

template<class _Ty> inline constexpr bool is_reference_v = type_traits::is_reference<_Ty>::value;

static_assert(!::is_reference_v<float>);
static_assert(::is_reference_v<const float&&>);
static_assert(::is_reference_v<volatile float&>);
static_assert(!::is_reference_v<const short*>);
static_assert(::is_reference_v<volatile std::string&&>);

static_assert(std::is_same_v<type_traits::add_lvalue_reference_t<float>, float&>);
static_assert(!std::is_same_v<type_traits::add_lvalue_reference_t<const float>, float&>);
static_assert(!std::is_same_v<type_traits::add_lvalue_reference_t<const volatile float&&>, float&>);
static_assert(std::is_same_v<type_traits::add_lvalue_reference_t<double&>, double&>);
static_assert(std::is_same_v<type_traits::add_lvalue_reference_t<const std::wstring&&>, const std::wstring&>);
static_assert(std::is_same_v<type_traits::add_lvalue_reference_t<const float*>, const float*&>);
