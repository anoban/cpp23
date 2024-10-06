#include <cstdlib>
#include <sstring>
#include <type_traits>

#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#define __SSTRING_NO_MOVE_SEMANTICS__

class object final {
        unsigned _value;

    public:
        object() noexcept                         = default;
        object(const object&) noexcept            = default;
        object& operator=(const object&) noexcept = default;
        ~object() noexcept                        = default;

        object(object&&)                          = delete;
        object& operator=(object&&)               = delete;
};

static_assert(std::is_standard_layout_v<object>);

// http://www.gotw.ca/publications/mill22.htm

static inline object factory() throw(std::is_nothrow_constructible_v<object>) { return object {}; }

static inline object _factory() throw(std::is_nothrow_constructible_v<object>) {
    object temporary {};
    return temporary;
}

int main() {
    const auto obj = ::factory();
    return EXIT_SUCCESS;
}
