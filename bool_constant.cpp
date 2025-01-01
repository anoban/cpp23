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
