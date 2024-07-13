template<typename T> class base {
    protected:
        T _wrapped;

    public:
        constexpr explicit base(const T& init) noexcept : _wrapped { init } { }

        constexpr T& unwrap() noexcept { return _wrapped; }

        constexpr const T& unwrap() const noexcept { return _wrapped; }
};

template<typename T> class derived : public base<T> {
    private:
        using derived::base::_wrapped;
        float _dummy;

    public:
        constexpr derived() noexcept : derived::base {}, _dummy {} { }

        constexpr derived(const T& init, const float& x) noexcept : derived::base { init }, _dummy { x } { }

        // using derived::base::unwrap;
};

auto wmain() -> int {
    constexpr auto constd {
        derived { 762, 9.97584 }
    };

    auto whats_inside { constd.unwrap() };
    // constd.unwrap()++;

    derived nonconst { 7.86543, 8.46 };

    auto  whats_in_there { nonconst.unwrap() };
    auto& ref { nonconst.unwrap() };
    ref *= 2.0;
    nonconst.unwrap()++;

    return 0;
}
