#include <type_traits>

template<typename _Ty, _Ty _init> struct integral_constant {
        static constexpr _Ty value = _init;
};

// using alias templates
template<bool _init> using bool_constant = integral_constant<bool, _init>;
using true_type                          = bool_constant<true>;
using false_type                         = bool_constant<false>;

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
static_assert(::true_type::value);
static_assert(!::false_type::value);
static_assert(inheritance::true_type::value);
static_assert(!inheritance::false_type::value);
