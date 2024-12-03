#include <cstdlib>
#include <numbers>
#include <sstring>
#include <type_traits>

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) class point final {
    public:
        typedef _Ty  value_type;
        typedef _Ty& reference;

    private:
        _Ty _x, _y, _z;

    public:
        inline point() noexcept;
        inline explicit point(_In_ const _Ty&) noexcept;
        inline point(_In_ const _Ty&, _In_ const _Ty&, _In_ const _Ty&) noexcept;
        inline point(_In_ const point&) noexcept;
        inline point(_In_ point&&) noexcept;
        inline reference operator=(_In_ const point&) noexcept;
        inline reference operator=(_In_ point&&) noexcept;
        inline ~point() noexcept;

        inline value_type x() const noexcept;
        inline value_type y() const noexcept;
        inline value_type z() const noexcept;

        inline reference  x() noexcept;
        inline reference  y() noexcept;
        inline reference  z() noexcept;

        inline void       x(_In_ const _Ty&) noexcept;
        inline void       y(_In_ const _Ty&) noexcept;
        inline void       z(_In_ const _Ty&) noexcept;
};

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline point<_Ty>::point() noexcept : _x(), _y(), _z() { }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>)
inline point<_Ty>::point(_In_ const _Ty& val) noexcept : _x(val), _y(val), _z(val) { }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>)
inline point<_Ty>::point(_In_ const _Ty& x, _In_ const _Ty& y, _In_ const _Ty& z) noexcept : _x(x), _y(y), _z(z) { }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>)
inline point<_Ty>::point(_In_ const point<_Ty>& other) noexcept : _x(other._x), _y(other._y), _z(other._z) { }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>)
inline point<_Ty>::point(_In_ point<_Ty>&& other) noexcept : _x(other._x), _y(other._y), _z(other._z) {
    other._x = other._y = other._z = _Ty(); // default initialize
}

template<class _Ty> requires(std::is_arithmetic_v<_Ty>)
inline point<_Ty>::reference point<_Ty>::operator=(_In_ const point<_Ty>& other) noexcept {
    if (this == std::addressof(other)) return *this;
    _x = other._x;
    _y = other._y;
    _z = other._z;
    return *this;
}

template<class _Ty> requires(std::is_arithmetic_v<_Ty>)
inline point<_Ty>::reference point<_Ty>::operator=(_In_ point<_Ty>&& other) noexcept {
    if (this == std::addressof(other)) return *this;
    _x       = other._x;
    _y       = other._y;
    _z       = other._z;

    other._x = other._y = other._z = _Ty();
    return *this;
}

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline point<_Ty>::~point() noexcept { _x = _y = _z = _Ty(); }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline point<_Ty>::value_type point<_Ty>::x() const noexcept { return _x; }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline point<_Ty>::value_type point<_Ty>::y() const noexcept { return _y; }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline point<_Ty>::value_type point<_Ty>::z() const noexcept { return _z; }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline point<_Ty>::reference  point<_Ty>::x() noexcept { return _x; }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline point<_Ty>::reference  point<_Ty>::y() noexcept { return _y; }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline point<_Ty>::reference  point<_Ty>::z() noexcept { return _z; }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline void                   point<_Ty>::x(_In_ const _Ty& x) noexcept { _x = x; }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline void                   point<_Ty>::y(_In_ const _Ty& y) noexcept { _y = y; }

template<class _Ty> requires(std::is_arithmetic_v<_Ty>) inline void                   point<_Ty>::z(_In_ const _Ty& z) noexcept { _z = z; }

template<class _Ty, class = typename std::enable_if<std::is_arithmetic<_Ty>::value, _Ty>::type> class rational final {
    public:
        using value_type = _Ty;
        using reference  = _Ty&;

    private:
        _Ty numerator, denominator;

    public:
        constexpr inline rational() noexcept;
        constexpr inline explicit rational(_In_ const _Ty&) noexcept;
        constexpr inline rational(_In_ const _Ty&, _In_ const _Ty&) noexcept;
};

template<class _Ty> inline rational<_Ty, typename std::enable_if<std::is_arithmetic<_Ty>::value, _Ty>::type>::rational() noexcept :
    numerator(), denominator() {};

int wmain() {
    // ::point<::sstring> invalid {};
    ::point valid { std::numbers::pi_v<double> };
    return EXIT_SUCCESS;
}
