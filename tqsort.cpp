#include <cstdint>
#include <type_traits>

template<typename T>
constexpr void swap(
    T* const __val0, T* const __val1, typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, T>::type = 0
) throw() {
    const T temp { *__val0 };
    *__val0 = *__val1;
    *__val1 = temp;
}

// overload using references
template<typename T, typename = std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, T>::type>
constexpr void swap(T& __val0, T& __val1) throw() {
    const T temp { __val0 };
    __val0 = __val1;
    __val1 = temp;
}

template<typename T>
constexpr std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, size_t>::type partition(
    T collection[], const size_t soffset, const size_t eoffset
) throw() {
    //
}

int main() {
    // a plain C style array
    int array[] { 37, 32, 12, 46, 38, 29, 45, 24, 32, 24, 3,  33, 33, 4,  30, 27, 17, 27, 13, 28, 6,  14, 16, 38, 26, 6,  28, 45, 13,
                  22, 26, 43, 18, 22, 29, 17, 16, 21, 29, 10, 11, 9,  20, 41, 36, 44, 6,  19, 22, 19, 4,  38, 22, 38, 39, 13, 22, 33,
                  40, 8,  20, 44, 16, 38, 20, 40, 11, 35, 42, 28, 36, 46, 19, 5,  7,  45, 45, 32, 30, 5,  47, 14, 5,  44, 24, 0,  42,
                  7,  8,  15, 12, 29, 34, 32, 43, 27, 5,  44, 18, 2,  2,  38, 3,  22, 29, 3,  0,  21, 8,  16, 43, 15, 8,  13, 22, 20,
                  49, 28, 0,  48, 3,  2,  0,  42, 23, 25, 7,  4,  6,  25, 41, 35, 34, 11, 7,  37, 4,  33, 3,  48, 3,  10, 48, 26, 30,
                  13, 3,  18, 15, 35, 6,  5,  24, 44, 8,  0,  14, 22, 23, 21, 17, 21, 26, 25, 20, 16, 4,  28, 39, 2,  1,  8,  35, 42,
                  36, 42, 26, 19, 24, 25, 17, 41, 19, 14, 29, 37, 25, 4,  24, 6,  5,  1,  45, 41, 49, 41, 41, 18, 22, 20 };

    return 0;
}
