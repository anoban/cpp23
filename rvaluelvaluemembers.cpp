#include <cstdlib>
#include <numbers>

class wrapper {
    private:
        float _value;

    public:
        constexpr wrapper() noexcept : _value {} { }

        constexpr explicit wrapper(const float& val) noexcept : _value { val } { }

        constexpr inline auto unwrap() & noexcept -> float& { return _value; } // overload usable for lvalues only

        constexpr inline auto unwrap() const& noexcept -> float { return std::numbers::pi; } // overload usable for lvalues only

        constexpr inline auto unwrap() && noexcept -> float& { return _value; } // overload usable for rvalues only

        constexpr inline auto unwrap() const&& noexcept -> float { return std::numbers::pi; } // overload usable for rvalues only

        wrapper(const wrapper&)            = delete;
        wrapper(wrapper&&)                 = delete;
        wrapper& operator=(const wrapper&) = delete;
        wrapper& operator=(wrapper&&)      = delete;
        constexpr ~wrapper() noexcept      = default;
};

constexpr auto dummy { 7.24564F };

auto           wmain() -> int {
    //
    constexpr auto pi = wrapper { dummy }.unwrap(); // 3.1415927F
    constexpr auto wrapped { wrapper(dummy) };
    constexpr auto unwrapped { wrapped.unwrap() }; // 7.24564F

    wrapper(std::numbers::inv_pi).unwrap() += 10.0000;
    const auto&& const_rvalue { wrapper(std::numbers::sqrt3) };
    const auto   whatsthat = const_rvalue.unwrap();

    auto         mutablewrapper { wrapper { std::numbers::egamma } };
    mutablewrapper.unwrap() *= 2.000;

    return EXIT_SUCCESS;
}
