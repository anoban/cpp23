#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

template<bool, typename T> struct enable_if {
    static const bool value = false;
};

template<typename T> struct enable_if<true, T> {
    typedef T type;
    static const bool value = true;
};

template<class T> struct is_arithmetic {
    typedef T type;
    static const bool value = false;
};

template<> struct is_arithmetic<char> {
    typedef char type;
    static const bool value = true;
};

template<> struct is_arithmetic<unsigned char> {
    typedef unsigned char type;
    static const bool value = true;
};

template<> struct is_arithmetic<short> {
    typedef short type;
    static const bool value = true;
};

template<> struct is_arithmetic<unsigned short> {
    typedef unsigned short type;
    static const bool value = true;
};

template<> struct is_arithmetic<int> {
    typedef int type;
    static const bool value = true;
};

template<> struct is_arithmetic<unsigned int> {
    typedef unsigned int type;
    static const bool value = true;
};

template<> struct is_arithmetic<long> {
    typedef long type;
    static const bool value = true;
};

template<> struct is_arithmetic<unsigned long> {
    typedef unsigned long type;
    static const bool value = true;
};

template<> struct is_arithmetic<long long> {
    typedef long long type;
    static const bool value = true;
};

template<> struct is_arithmetic<unsigned long long> {
    typedef unsigned long long type;
    static const bool value = true;
};

template<> struct is_arithmetic<float> {
    typedef float type;
    static const bool value = true;
};

template<> struct is_arithmetic<double> {
    typedef double type;
    static const bool value = true;
};

template<> struct is_arithmetic<long double> {
    typedef long double type;
    static const bool value = true;
};

template<typename T, typename U> static typename ::enable_if<::is_arithmetic<T>::value, long double>::type
sum(const T& arg_0, const U& arg_1) throw(){
    return arg_0 + arg_1;
}

int main(){
    static const long double twopi = ::sum(M_PI, M_PI);
    std::cout << twopi << std::endl;
    ::getchar();
    return EXIT_SUCCESS;
}