#include <memory>
#include <type_traits>

template<class _Ty> class unique_ptr final {
    public:
        using value_type    = _Ty;
        using pointer       = _Ty*;
        using const_pointer = const _Ty*;
        using size_type     = unsigned long long;

    private:
        pointer   _resource;
        size_type _size;

    public:
        unique_ptr() noexcept : _resource(::new (std::nothrow) _Ty), _size(1) { }

        // WITH ::new WE CAN MAINTAN TYPE CONSISTENCY
        // explicit unique_ptr(_In_ const size_type& _size) noexcept : _resource(::new (std::nothrow) _Ty[_size]), _size(_size) { }

        // IF WE OPT TO USE ::malloc WE WILL RUN INTO TYPE ERRORS WHEN _Ty IS NOT void
        explicit unique_ptr(_In_ const size_type& _size) noexcept : _resource(::malloc(sizeof(_Ty) * _size)), _size(_size) { }

        unique_ptr(const unique_ptr&) noexcept            = delete;

        unique_ptr(unique_ptr&&) noexcept                 = delete;

        unique_ptr& operator=(const unique_ptr&) noexcept = delete;

        unique_ptr& operator=(unique_ptr&&) noexcept      = delete;

        ~unique_ptr() noexcept {
            _size == 1 ? delete _resource : delete[] _resource;
            _resource = nullptr;
            _size     = 0;
        }

        pointer get() noexcept { return _resource; }

        const_pointer get() const noexcept { return _resource; }
};

auto wmain() -> int {
    auto hundred { ::unique_ptr<void>(100) };

    return EXIT_SUCCESS;
}
