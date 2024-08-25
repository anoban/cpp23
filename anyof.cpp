#include <concepts>
#include <type_traits>

// make sure the passed predicate template has a public static const bool memeber named value
template<template<class> class predicate, class candidate> concept validate_predicate =
    std::is_same<class predicate<candidate>::value, bool>::value;

namespace recursive {
    template<template<class> class predicate, class T, class... TList> struct any_of final {
            static constexpr bool value = predicate<T>::value || any_of<predicate, TList...>::value;
    };

    template<template<class> class predicate, class T> struct any_of<predicate, T> final {
            static constexpr bool value = predicate<T>::value;
    };
} // namespace recursive

static_assert(recursive::any_of<std::is_floating_point, float, double, short, unsigned, int>::value);
static_assert(!recursive::any_of<std::is_floating_point, char, long long, short, unsigned, int>::value);

namespace fold_expressions {

    template<template<typename> typename predicate, typename... TList> struct any_of final {
            static constexpr bool value = (predicate<TList>::value || ...);
    };

    template<template<typename> typename predicate, typename... TList> struct all_of final {
            static constexpr bool value = (predicate<TList>::value && ...);
    };

} // namespace fold_expressions

static_assert(fold_expressions::any_of<std::is_floating_point, float, double, short, unsigned, int>::value);
static_assert(!fold_expressions::any_of<std::is_floating_point, char, long long, short, unsigned, int, long>::value);
static_assert(fold_expressions::all_of<std::is_floating_point, float, double, long double>::value);
static_assert(!fold_expressions::all_of<std::is_floating_point, char, float, double, long double>::value);

namespace constexprif { // still recursive though

    template<template<typename> typename predicate, typename T, typename... TList> struct any_of final {
            static constexpr bool operator()() noexcept {
                if constexpr (!sizeof...(TList)) // when TList is empty
                    return predicate<T>::value;
                else
                    return predicate<T>::value || any_of<predicate, TList...>::operator()();
            }
    };

    template<template<typename> typename predicate, typename T, typename... TList> struct all_of final {
            static constexpr bool operator()() noexcept {
                if constexpr (!sizeof...(TList)) // when TList is empty
                    return predicate<T>::value;
                else
                    return predicate<T>::value && all_of<predicate, TList...>::operator()();
            }
    };

} // namespace constexprif

static_assert(constexprif::any_of<std::is_floating_point, float, double, short, unsigned, int>::operator()());
static_assert(!constexprif::any_of<std::is_floating_point, char, long long, short, unsigned, int, long>::operator()());
static_assert(constexprif::all_of<std::is_floating_point, float, double, long double>::operator()());
static_assert(!constexprif::all_of<std::is_floating_point, char, float, double, long double>::operator()());
