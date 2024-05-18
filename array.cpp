#include <cstddef>
#include <cstdlib>
#include <numbers>
#include <type_traits>

#include <sal.h>

template<typename scalar_t, size_t _length, typename = std::enable_if<std::is_scalar<scalar_t>::value, scalar_t>::type>
class random_access_iterator { };

template<typename scalar_t, size_t _length, typename = std::enable_if<std::is_scalar<scalar_t>::value, scalar_t>::type> class array {
        using value_type   = scalar_t;
        using pointer_type = scalar_t*;
        using iterator     = random_access_iterator<value_type, _length>;
        // using const_iterator = ;

    public:
        constexpr array() noexcept : _buffer {}, _size { _length } { }

        constexpr explicit array(_In_ const value_type _val) noexcept : _buffer { _val }, _size { _length } { }

        constexpr explicit array(_In_ value_type _val, _In_ const size_t _count) noexcept : _size { _length } {
            size_t i {};
            for (; i < _count; ++i) _buffer[i] = _val;
            for (; i < _size; ++i) _buffer[i] = static_cast<value_type>(0);
        }

        constexpr size_t size() const noexcept { return _size; }

        constexpr bool is_empty() const noexcept { return _size != 0u; }

        constexpr pointer_type data() const noexcept { return _buffer; }

    private:
        scalar_t _buffer[_length]; // avoiding in-class initializers for perfromance
        size_t   _size;
};

int wmain() {
    constexpr auto x { ::array<short, 2000> {} };
    constexpr auto y { ::array<float, 100> { std::numbers::pi_v<float> } };
    constexpr auto z {
        ::array<double, 100> { std::numbers::egamma_v<double>, 50 }
    };

    return EXIT_SUCCESS;
}
