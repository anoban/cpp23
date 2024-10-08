#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <cstdlib>
#include <sstring>
#include <type_traits>

class copy_only_object final {
        unsigned _value;

    public:
        copy_only_object() noexcept                                   = default;

        copy_only_object(const copy_only_object&) noexcept            = default;
        copy_only_object& operator=(const copy_only_object&) noexcept = default;

        copy_only_object(copy_only_object&&) noexcept                 = delete;
        copy_only_object& operator=(copy_only_object&&) noexcept      = delete;

        ~copy_only_object() noexcept                                  = default;
};

static_assert(std::is_standard_layout_v<copy_only_object>);

// http://www.gotw.ca/publications/mill22.htm

static inline copy_only_object factory() noexcept { return copy_only_object {}; }

static inline copy_only_object _factory() noexcept {
    copy_only_object temporary {};
    return temporary; // requires a valid move constructor
}

int main() {
    const auto obj = ::factory();
    return EXIT_SUCCESS;
}
