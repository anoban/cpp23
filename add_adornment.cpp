#include <cstdlib>
#include <numbers>
#include <type_traits>

static_assert(std::is_same_v<std::add_const_t<float>, const float>);
static_assert(!std::is_same_v<std::add_const_t<float>, const float&>);

static_assert(std::is_same_v<std::add_const_t<const float&>, const float&>); // redundant type qualifiers are ignored in C++ TMP

static constexpr auto pi { std::numbers::pi_v<double> };

static_assert(std::is_same_v<std::add_const_t<double>, decltype(pi)>);

auto wmain() {
    // mighty intersting huh :)
    return EXIT_SUCCESS;
}
