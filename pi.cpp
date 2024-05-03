#include <concepts>

template<class T>
static inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type pi = static_cast<T>(3.14159265358979L);

int main() {
    constexpr auto value { ::pi<float> };
    constexpr auto ivalue { ::pi<int16_t> };
    return 0;
}
