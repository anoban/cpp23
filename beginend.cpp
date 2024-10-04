#include <cstdlib>
#include <type_traits>

namespace nstd {

    template<class T, size_t length> constexpr T* begin(T (&array)[length]) noexcept { return array; }

    template<class T, size_t length> constexpr T* end(T (&array)[length]) noexcept { return array + length; }

} // namespace nstd

static char text[] { "char array" };

static_assert(std::is_same_v<decltype(nstd::begin(text)), char*>);
static_assert(std::is_same_v<decltype(nstd::begin("string literal")), const char*>);

auto wmain() -> int {
    //
    return EXIT_SUCCESS;
}
