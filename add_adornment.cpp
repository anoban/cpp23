#include <type_traits>

static_assert(std::is_same_v<std::add_const_t<float>, const float>);
static_assert(!std::is_same_v<std::add_const_t<float>, const float&>);

static_assert(std::is_same_v<std::add_const_t<const float&>, const float&>); // redundant qualifiers are ignored in C++ TMP
