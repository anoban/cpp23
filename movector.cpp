#include <cstdlib>
#include <type_traits>

class object final {
        unsigned _value;

    public:
        constexpr object() noexcept               = default;
        object(const object&) noexcept            = default;
        object& operator=(const object&) noexcept = default;
        ~object() noexcept                        = default;

        object(object&&)                          = delete;
        object& operator=(object&&)               = delete;
};

static_assert(std::is_standard_layout_v<object>);

static inline constexpr object factory() noexcept(std::is_nothrow_constructible_v<object>) { return object {}; }

static inline constexpr object _factory() noexcept(std::is_nothrow_constructible_v<object>) {
    object temporary {};
    return temporary;
}

auto main() -> int {
    //
    constexpr auto obj { ::factory() };
    return EXIT_SUCCESS;
}
