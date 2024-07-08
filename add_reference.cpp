#include <type_traits>

namespace __partial_specializations {

    template<typename T> struct add_reference {
            using type = T&;
    };

    // consider adding specializations for void types
    template<> struct add_reference<void> {
            using type = void;
    };

    template<> struct add_reference<const void> {
            using type = const void;
    };

    template<> struct add_reference<volatile void> {
            using type = volatile void;
    };

    template<> struct add_reference<const volatile void> {
            using type = const volatile void;
    };

    template<typename T> using add_reference_t = typename add_reference<T>::type;

    static_assert(std::is_same_v<add_reference_t<float>, float&>);
    static_assert(std::is_same_v<add_reference_t<const float>, const float&>);
    static_assert(std::is_same_v<add_reference_t<volatile float>, volatile float&>);
    static_assert(std::is_same_v<add_reference_t<const volatile float>, const volatile float&>);
    static_assert(std::is_same_v<add_reference_t<float*>, float*&>);
    static_assert(std::is_same_v<add_reference_t<short&&>, short&>);
    static_assert(std::is_same_v<add_reference_t<short&>, short&>);

    // what will happen when T is void?
    static_assert(std::is_same_v<add_reference_t<void>, void>);                               // okay because the specialization saved us
    static_assert(std::is_same_v<add_reference_t<const void>, const void>);                   // okay because the specialization saved us
    static_assert(std::is_same_v<add_reference_t<volatile void>, volatile void>);             // okay because the specialization saved us
    static_assert(std::is_same_v<add_reference_t<const volatile void>, const volatile void>); // okay because the specialization saved us

} // namespace __partial_specializations

namespace __type_constraints {

    // let's reimplement add_reference using type constraints

    namespace __requires {
        // using requires will help us filter out void types but will still give us a hard error if a void type is provided
        template<typename T> requires(!std::is_void_v<T>) struct add_reference {
                using type = T&;
        };

        template<typename T> using add_reference_t = typename add_reference<T>::type;

        static_assert(std::is_same_v<add_reference_t<float>, float&>);
        static_assert(std::is_same_v<add_reference_t<const float>, const float&>);
        static_assert(std::is_same_v<add_reference_t<volatile float>, volatile float&>);
        static_assert(std::is_same_v<add_reference_t<const volatile float>, const volatile float&>);
        static_assert(std::is_same_v<add_reference_t<float*>, float*&>);
        static_assert(std::is_same_v<add_reference_t<short&&>, short&>);
        static_assert(std::is_same_v<add_reference_t<short&>, short&>);

        // what will happen when T is void?
        static_assert(std::is_same_v<add_reference_t<void>, void>);                               // type constraint is not satisfied!
        static_assert(std::is_same_v<add_reference_t<const void>, const void>);                   // type constraint is not satisfied!
        static_assert(std::is_same_v<add_reference_t<volatile void>, volatile void>);             // type constraint is not satisfied!
        static_assert(std::is_same_v<add_reference_t<const volatile void>, const volatile void>); // type constraint is not satisfied!

        // but we'd rather not have hard errors

    } // namespace __requires

    namespace __conditional_template_predicate {
        template<typename T, bool is_void = std::is_void_v<T>> struct add_reference {
                using type = T&;
        };

        // a partial specialization for cases when T is  void type
        template<typename T> struct add_reference<T, true> {
                using type = T;
        };

        template<typename T> using add_reference_t = typename add_reference<T>::type;

        static_assert(std::is_same_v<add_reference_t<float>, float&>);
        static_assert(std::is_same_v<add_reference_t<const float>, const float&>);
        static_assert(std::is_same_v<add_reference_t<volatile float>, volatile float&>);
        static_assert(std::is_same_v<add_reference_t<const volatile float>, const volatile float&>);
        static_assert(std::is_same_v<add_reference_t<float*>, float*&>);
        static_assert(std::is_same_v<add_reference_t<short&&>, short&>);
        static_assert(std::is_same_v<add_reference_t<short&>, short&>);

        // what will happen when T is void?
        static_assert(std::is_same_v<add_reference_t<void>, void>);                   // partial specialization to the rescue
        static_assert(std::is_same_v<add_reference_t<const void>, const void>);       // partial specialization to the rescue
        static_assert(std::is_same_v<add_reference_t<volatile void>, volatile void>); // partial specialization to the rescue
        static_assert(std::is_same_v<add_reference_t<const volatile void>,
                                     const volatile void>); // partial specialization to the rescue

    } // namespace __conditional_template_predicate

} // namespace __type_constraints

namespace __arthur_o_dwyer__ {

    template<class T, class SFINAE> struct impl {
            using type = T;
    };

    template<class T> struct impl<T, std::remove_reference_t<T&>> { // a partial specialization where T& must be well formed
            using type = T&;
    };

    // when T& is ill formed, the compiler will fall back to the base template

    template<typename T> using add_reference_t = typename impl<T, T>::type;
    // the problem with using = typename impl<T, T>::type; is rvalue references
    // when T is an rvalue reference, our partial specialization is goint to attempt to form a T&&& in
    // struct impl<T, std::remove_reference_t<T&>>, that is ill-formed
    // the compiler will fall back to use the base template which we do not want

    static_assert(std::is_same_v<add_reference_t<float>, float&>);
    static_assert(std::is_same_v<add_reference_t<const float>, const float&>);
    static_assert(std::is_same_v<add_reference_t<volatile float>, volatile float&>);
    static_assert(std::is_same_v<add_reference_t<const volatile float>, const volatile float&>);
    static_assert(std::is_same_v<add_reference_t<float*>, float*&>);
    static_assert(std::is_same_v<add_reference_t<short&&>, short&>);
    static_assert(std::is_same_v<add_reference_t<short&>, short&>);

    // what will happen when T is void?
    static_assert(std::is_same_v<add_reference_t<void>, void>);                   // base template to the rescue
    static_assert(std::is_same_v<add_reference_t<const void>, const void>);       // base template to the rescue
    static_assert(std::is_same_v<add_reference_t<volatile void>, volatile void>); // base template to the rescue
    static_assert(std::is_same_v<add_reference_t<const volatile void>,
                                 const volatile void>); // base template to the rescue

} // namespace __arthur_o_dwyer__

static_assert(std::is_same_v<std::add_lvalue_reference_t<short&&>, short&>);
