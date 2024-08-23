// references and temporaries
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <type_traits>

template<bool predicate, class T> struct enable_if { };

template<class T> struct enable_if<true, T> {
        static const bool value = true;
        typedef T         type;
};

template<class T> static __declspec(noinline) ::enable_if<std::is_arithmetic<T>::value, T>::type function(_In_ const T& x) noexcept {
    return M_PI;
}

int main() {
    const auto pi = ::function('A');

    // reference to temporaries
    return EXIT_SUCCESS;
}
