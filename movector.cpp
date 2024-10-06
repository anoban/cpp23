#include <cstdlib>
#include <sstring>
#include <type_traits>

#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
// #define __SSTRING_NO_MOVE_SEMANTICS__

class object final {
        unsigned _value;

    public:
        object() noexcept                         = default;
        object(const object&) noexcept            = default;
        object& operator=(const object&) noexcept = default;
        ~object() noexcept                        = default;

        object(object&&) noexcept                 = delete;
        object& operator=(object&&) noexcept      = delete;
};

static_assert(std::is_standard_layout_v<object>);

// http://www.gotw.ca/publications/mill22.htm

static inline object factory() noexcept { return object {}; }

static inline object _factory() noexcept {
    object temporary {};
    return temporary;
}

int main() {
    const auto obj = ::factory();
    return EXIT_SUCCESS;
}
