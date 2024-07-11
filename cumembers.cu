#include <concepts>

template<typename T> requires std::is_scalar_v<T> class object { };
