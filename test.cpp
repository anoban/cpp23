#include <cstdalign>
#include <cstddef>
#include <cstdint>
#include <numbers>

template<unsigned eval> static unsigned evaluate() noexcept { return eval; }

struct align {
        bool      first;
        int16_t   second;
        double    third;
        char      fourth;
        long long fifth;
};

struct __declspec(align(128)) align_ {
        bool      first;
        int16_t   second;
        double    third;
        char      fourth;
        long long fifth;
        float     sixth;
};

template<unsigned value_0, unsigned value_1> struct product {
        static const unsigned result { value_0 * value_1 };
};

// following template parameters can depend on previous parameters, as value_0 and value_1 depend on the type scalar_t here
template<typename scalar_t, scalar_t value_0, scalar_t value_1> struct product_v {
        static const scalar_t result { value_0 * value_1 };
        // for float types, result must be constexpr
};

// class types can be templated
template<typename T> class TClass {
    private:
        T mvar {};
};

// a templated class type can have templated member functions whose template arguments may or may not depend on the class templates' arguments
template<typename T, typename U, T t_scalar, U u_scalar> class TUClass {
    public:
        static const T value_t { t_scalar };
        static const U value_u { u_scalar };

        // a member function dependent on the class template's arguments
        static constexpr T value() noexcept { return value_t; } // return type T is a type argument of class TUClass

        // a member function using new templates
        template<typename X> constexpr X func(const X one, const X two) noexcept { return one + two; }

        // using the same template argument name as the parent template does not imply that the member's type is the parent's type
        template<typename T /* this T will be inferred independently from class template arguments */>
        constexpr T funcV2(const T one, const T two) noexcept {
            return static_cast<T>(one + two);
        }
};

// compile time constants can be...
int main() {
    const unsigned             cu { 71 };
    constexpr long             cexpl { 14 };
    static const unsigned char scuc { 148 };

    evaluate<sizeof(double)>(); // okay
    evaluate<alignof(align::fifth)>();
    evaluate<alignof(align_)>();
    evaluate<offsetof(align_, fourth)>();
    evaluate<19>();
    evaluate<__LINE__>();

    evaluate<cu>();
    evaluate<cexpl>();
    evaluate<scuc>();

    auto       hundred { product<25, 4>::result };
    auto       fifty { product_v<10.00L, 5.000L>::result };       // Error: missing template argument typename scalar_t
    auto       _fifty { product_v<float, 10.00, 5.000>::result }; // Error: truncation from double -> float in template arguments
    auto       fifty_ { product_v<float, 10.00F, 5.000F>::result };
    const auto thirty { product_v<int, 15, 2>::result };

    auto okay { product_v<double, 1867.6540, 97565.565>::result };

    // an explicit cast helps with template arguments
    auto __fifty { product_v<float, static_cast<float>(10.00), static_cast<float>(5.000L)>::result };

    evaluate<product<7, 6>::result>();

    auto       object { TUClass<float, long, 6.67457F, 12> {} };
    const auto pi { TUClass<float, long, std::numbers::pi_v<float>, 12>::value_t };

    const auto x { object.func('A', 'X') }; // templated member with arguments independent of the class template

    const auto v2 { object.funcV2(45I16, 6753I16) };

    return 0;
}
