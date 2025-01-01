#include <type_traits>

// using alias templates
namespace alias {

    template<typename _Ty, _Ty _init> struct integral_constant final {
            static constexpr _Ty value = _init;
    };

    template<bool _init> using bool_constant = integral_constant<bool, _init>;
    using true_type                          = bool_constant<true>;
    using false_type                         = bool_constant<false>;

}

// using inheritance
namespace inheritance {

    template<typename _Ty, _Ty _init> struct integral_constant {
            static constexpr _Ty value = _init;
    };

    template<bool _init> struct bool_constant final : integral_constant<bool, _init> { };

    using true_type  = bool_constant<true>;
    using false_type = bool_constant<false>;

};

// making progress :)
static_assert(alias::true_type::value);
static_assert(!alias::false_type::value);
static_assert(inheritance::true_type::value);
static_assert(!inheritance::false_type::value);

template<typename _Ty> struct remove_reference final {
        using type = _Ty;
};

template<typename _Ty> struct remove_reference<_Ty&> final {
        using type = _Ty;
};

template<typename _Ty> struct remove_reference<_Ty&&> final {
        using type = _Ty;
};

static_assert(std::is_same_v<double, ::remove_reference<double&&>::type>);
static_assert(std::is_same_v<const volatile float, ::remove_reference<const volatile float&>::type>);
static_assert(std::is_same_v<volatile unsigned long, ::remove_reference<volatile unsigned long>::type>);

template<typename _Ty> struct add_lvalue_reference final {
        using type = _Ty&;
};

template<> struct add_lvalue_reference<void> final {
        using type = void;
};

template<> struct add_lvalue_reference<const void> final {
        using type = void;
};

template<> struct add_lvalue_reference<volatile void> final {
        using type = void;
};

template<> struct add_lvalue_reference<const volatile void> final {
        using type = void;
};

namespace alternative_00 {

    template<typename _Ty, typename _TyDuplicate> struct _lvalue_reference_helper final {
            using type = _Ty&;
    };

    template<typename _Ty> struct _lvalue_reference_helper<_Ty, void> final {
            using type = _Ty;
    };

    template<typename _Ty> using add_lvalue_reference_t = typename _lvalue_reference_helper<_Ty, std::remove_cv<_Ty>::type>::type;

}

static_assert(std::is_same_v<void, alternative_00::add_lvalue_reference_t<void>>);
static_assert(std::is_same_v<const void, alternative_00::add_lvalue_reference_t<const void>>);
static_assert(std::is_same_v<volatile void, alternative_00::add_lvalue_reference_t<volatile void>>);
static_assert(std::is_same_v<const volatile void, alternative_00::add_lvalue_reference_t<const volatile void>>);
static_assert(std::is_same_v<float&, alternative_00::add_lvalue_reference_t<float&&>>);
static_assert(std::is_same_v<const float&, alternative_00::add_lvalue_reference_t<const float>>);
static_assert(std::is_same_v<volatile long long&, alternative_00::add_lvalue_reference_t<volatile long long&>>);

namespace alternative_01 {

    template<typename _Ty, bool _is_void> struct _lvalue_reference_helper final {
            using type = _Ty&;
    };

    template<typename _Ty> struct _lvalue_reference_helper<_Ty, true> final {
            using type = _Ty;
    };

    template<typename _Ty> using add_lvalue_reference_t = typename _lvalue_reference_helper<_Ty, std::is_void<_Ty>::value>::type;

}

static_assert(std::is_same_v<void, alternative_01::add_lvalue_reference_t<void>>);
static_assert(std::is_same_v<const void, alternative_01::add_lvalue_reference_t<const void>>);
static_assert(std::is_same_v<volatile void, alternative_01::add_lvalue_reference_t<volatile void>>);
static_assert(std::is_same_v<const volatile void, alternative_01::add_lvalue_reference_t<const volatile void>>);
static_assert(std::is_same_v<float&, alternative_01::add_lvalue_reference_t<float&&>>);
static_assert(std::is_same_v<const float&, alternative_01::add_lvalue_reference_t<const float>>);
static_assert(std::is_same_v<volatile long long&, alternative_01::add_lvalue_reference_t<volatile long long&>>);

namespace alternative_02 {

    template<typename _Ty, bool _is_void> struct _lvalue_reference_helper final {
            using type = _Ty&;
    };

    template<typename _Ty> struct _lvalue_reference_helper<_Ty, true> final {
            using type = _Ty;
    };

    template<typename _Ty> using add_lvalue_reference_t = typename _lvalue_reference_helper<_Ty, std::is_void<_Ty>::value>::type;

}
