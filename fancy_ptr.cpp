template<typename T> class unique_ptr final {
    private:
        T* _resource {}; // pointer to the raw heap resource

    public:
        using value_type      = T;
        using pointer         = T*;
        using const_pointer   = const T*;
        using reference       = T&;
        using const_reference = const T&;
        using difference_type = signed long long;
        using size_type       = unsigned long long;

        constexpr unique_ptr() noexcept : _resource { nullptr } { }

        constexpr explicit unique_ptr(T* ptr) noexcept : _resource { ptr } { }
};
