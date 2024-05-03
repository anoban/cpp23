#include <concepts>

template<class T>
static inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type pi = static_cast<T>(3.14159265358979L);

template<typename T> struct is_arithmetic;

template<> struct is_arithmetic<char> {
        typedef char          type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<unsigned char> {
        typedef unsigned char type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<short> {
        typedef short         type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<unsigned short> {
        typedef unsigned short type;
        static constexpr bool  value { true };
};

template<> struct is_arithmetic<int> {
        typedef int           type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<unsigned int> {
        typedef unsigned int  type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<long> {
        typedef long          type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<unsigned long> {
        typedef unsigned long type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<long long> {
        typedef long long     type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<unsigned long long> {
        typedef unsigned long long type;
        static constexpr bool      value { true };
};

template<> struct is_arithmetic<float> {
        typedef float         type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<double> {
        typedef double        type;
        static constexpr bool value { true };
};

template<> struct is_arithmetic<long double> {
        typedef long double   type;
        static constexpr bool value { true };
};

template<typename T> constexpr bool is_arithmetic_v = ::is_arithmetic<T>::value;

template<typename T> typename is_arithmetic_t       = ::is_arithmetic<T>::type;

int main() {
    constexpr auto value { ::pi<float> };
    constexpr auto ivalue { ::pi<int16_t> };

    constexpr auto is_it   = ::is_arithmetic_v<double>;
    constexpr auto it_isnt = ::is_arithmetic_v<bool>;

    return 0;
}
