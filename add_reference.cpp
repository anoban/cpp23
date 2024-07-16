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
        static_assert(std::is_same_v<add_reference_t<void>, void>);                               // partial specialization to the rescue
        static_assert(std::is_same_v<add_reference_t<const void>, const void>);                   // partial specialization to the rescue
        static_assert(std::is_same_v<add_reference_t<volatile void>, volatile void>);             // partial specialization to the rescue
        static_assert(std::is_same_v<add_reference_t<const volatile void>, const volatile void>); // partial specialization to the rescue

    } // namespace __conditional_template_predicate

} // namespace __type_constraints

namespace __arthur_o_dwyer {

    template<class T, class SFINAE> struct impl { // base template for SFINAE failures
            using type = T;
    };

    template<class T> struct impl<T, std::remove_reference_t<T&>> { // a partial specialization where T& must be well formed
            using type = T&;
    };

    // when T& is ill formed, the compiler will fall back to the base template

    template<typename T> using add_reference_t = typename impl<T, T>::type;
    // the problem with using = typename impl<T, T>::type; is reference types
    // when T is a reference, our result should always be T& (regardless of the type of reference T)
    // with add_reference_t<T&&>, our partial specialization will map to impl<T&&, T&&>
    // is the second type identical to the reference removed version of the first? (T&& == T) ? NO
    // with add_reference_t<T&>, our partial specialization will map to impl<T&, T&>
    // is the second type identical to the reference removed version of the first? (T& == T) ? NO
    // hence the compiler will fall back to use the base template, and our result will be the type of the input (T&& and T&)

    // refernce collapsing examples
    template<typename T, typename type = std::remove_reference<T&>::type> struct refer_it {
            using reference_type = T&;
    };

    static_assert(std::is_same_v<refer_it<float>::reference_type,
                                 float&>); // collapsed type is float&
    static_assert(std::is_same_v<refer_it<float&>::reference_type, float&>);
    static_assert(std::is_same_v<refer_it<float&&>::reference_type, float&>);

    // one workaround is
    template<typename T> using __add_reference_t = typename impl<T, std::remove_reference_t<T>>::type;

    static_assert(std::is_same_v<add_reference_t<float>, float&>);
    static_assert(std::is_same_v<add_reference_t<const float>, const float&>);
    static_assert(std::is_same_v<add_reference_t<volatile float>, volatile float&>);
    static_assert(std::is_same_v<add_reference_t<const volatile float>, const volatile float&>);
    static_assert(std::is_same_v<add_reference_t<float*>, float*&>);
    static_assert(std::is_same_v<add_reference_t<short&&>, short&>); // fall back
    static_assert(std::is_same_v<add_reference_t<short&>, short&>);  // fall back

    static_assert(std::is_same_v<__add_reference_t<short&&>, short&>); // okay :)
    static_assert(std::is_same_v<__add_reference_t<short&>, short&>);  // okay :)
    static_assert(std::is_same_v<__add_reference_t<void>, void>);      // okay :)

    // what will happen when T is void?
    static_assert(std::is_same_v<add_reference_t<void>, void>);                               // base template to the rescue
    static_assert(std::is_same_v<add_reference_t<const void>, const void>);                   // base template to the rescue
    static_assert(std::is_same_v<add_reference_t<volatile void>, volatile void>);             // base template to the rescue
    static_assert(std::is_same_v<add_reference_t<const volatile void>, const volatile void>); // base template to the rescue

} // namespace __arthur_o_dwyer

namespace __punned_types {
    // C++ has a new way to default a variadic type list to a single type
    // FOR WHAT THOUGH?

    template<typename...> using make_void = void; // our type of choice here is void

    static_assert(std::is_same_v<make_void<float, const int, volatile double, unsigned short&, const volatile char>, void>); // ;)
    // this will still not work with invalid types like void&

    template<typename T, typename SFINAE> struct impl { // base template, fall back for void types
            using type = T;
    };

    template<typename T> struct impl<T, make_void<T&>> {
            using type = T&;
    };

    // now that we have a template that needs two arguments, we need an expansion template (make two template arguments from one)
    template<typename T> struct add_lvalue_reference {
            using reference_type = typename impl<T, void>::type;
    };

    template<typename T> using add_lvalue_reference_t = typename add_lvalue_reference<T>::reference_type;

    static_assert(std::is_same_v<add_lvalue_reference_t<long>, long&>);
    static_assert(std::is_same_v<add_lvalue_reference_t<const float>, const float&>);
    static_assert(std::is_same_v<add_lvalue_reference_t<volatile float>, volatile float&>);
    static_assert(std::is_same_v<add_lvalue_reference_t<const volatile float>, const volatile float&>);
    static_assert(std::is_same_v<add_lvalue_reference_t<float*>, float*&>);
    static_assert(std::is_same_v<add_lvalue_reference_t<short&&>, short&>);
    static_assert(std::is_same_v<add_lvalue_reference_t<short&>, short&>);

    static_assert(std::is_same_v<add_lvalue_reference_t<void>, void>);                               // base template to the rescue
    static_assert(std::is_same_v<add_lvalue_reference_t<const void>, const void>);                   // base template to the rescue
    static_assert(std::is_same_v<add_lvalue_reference_t<volatile void>, volatile void>);             // base template to the rescue
    static_assert(std::is_same_v<add_lvalue_reference_t<const volatile void>, const volatile void>); // base template to the rescue

} // namespace __punned_types

namespace __experimenting_with_make_void {
    template<typename...> using make_void = void;

    template<typename T, typename SFINAE> struct impl {
            using reference_type = T;
    };

    template<typename T> struct impl<T, make_void<T&>> {
            using reference_type = T&&;
    };

    template<typename T> using add_rvalue_reference_t = typename impl<T, void>::reference_type;

    static_assert(std::is_same_v<add_rvalue_reference_t<long>, long&&>);
    static_assert(std::is_same_v<add_rvalue_reference_t<const float>, const float&&>);
    static_assert(std::is_same_v<add_rvalue_reference_t<volatile float>, volatile float&&>);
    static_assert(std::is_same_v<add_rvalue_reference_t<const volatile float>, const volatile float&&>);
    static_assert(std::is_same_v<add_rvalue_reference_t<float*>, float*&&>);
    static_assert(std::is_same_v<add_rvalue_reference_t<short&&>, short&&>);
    static_assert(std::is_same_v<add_rvalue_reference_t<short&>, short&>);

    static_assert(std::is_same_v<add_rvalue_reference_t<void>, void>);
    static_assert(std::is_same_v<add_rvalue_reference_t<const void>, const void>);
    static_assert(std::is_same_v<add_rvalue_reference_t<volatile void>, volatile void>);
    static_assert(std::is_same_v<add_rvalue_reference_t<const volatile void>, const volatile void>);

    template<typename T, typename SFINAE> struct point_to {
            using pointer_type = T;
    };

    template<typename T> struct point_to<T, make_void<T*>> {
            using pointer_type = T*;
    };

    template<typename T> struct add_pointer {
            using type = typename point_to<T, void>::pointer_type;
    };

    template<typename T> using add_pointer_t = typename add_pointer<T>::type;

    static_assert(std::is_same_v<add_pointer_t<long>, long*>);
    static_assert(std::is_same_v<add_pointer_t<const float>, const float*>);
    static_assert(std::is_same_v<add_pointer_t<volatile float>, volatile float*>);
    static_assert(std::is_same_v<add_pointer_t<const volatile float>, const volatile float*>);
    static_assert(std::is_same_v<add_pointer_t<float*>, float**>);
    static_assert(std::is_same_v<add_pointer_t<void>, void*>);
    static_assert(std::is_same_v<add_pointer_t<const void>, const void*>);
    static_assert(std::is_same_v<add_pointer_t<volatile void>, volatile void*>);
    static_assert(std::is_same_v<add_pointer_t<const volatile void>, const volatile void*>);

    static_assert(std::is_same_v<add_pointer_t<short&>, short&>);
    static_assert(std::is_same_v<add_pointer_t<short&&>, short&&>);
    static_assert(std::is_same_v<add_pointer_t<const double&&>, const double&&>);
    static_assert(std::is_same_v<add_pointer_t<volatile float&>, volatile float&>);
    static_assert(std::is_same_v<add_pointer_t<const volatile int&>, const volatile int&>);

} // namespace __experimenting_with_make_void
