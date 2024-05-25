#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <type_traits>

// a POD C style struct (aggregate type)
struct point3d {
        float x, y, z; // no inclass initializers
};

// in C, we'd need a helper to print point32 to the console or the user code will have to unpack the struct and print it using a regular *printf* family function
static void PrintPoint3D(const point3d& point) noexcept { wprintf_s(L"Point3D :: (%.4f, %.4f, %.4f)\n", point.x, point.y, point.z); }

static constexpr point3d NewPoint3D() noexcept { return point3d { 0, 0, 0 }; }

// in C++, point3d is likely to be implemented as a class type (non-aggregate)
class point3dcpp final {
    private:
        float x, y, z;

    public:
        constexpr point3dcpp() noexcept : x {}, y {}, z {} { }

        constexpr explicit point3dcpp(const float& _x) noexcept : x { _x }, y { _x }, z { _x } { }

        constexpr point3dcpp(const float& _x, const float& _y, const float& _z) noexcept : x { _x }, y { _y }, z { _z } { }

        ~point3dcpp() = default;

        constexpr float& getx() noexcept { return x; }

        constexpr float& gety() noexcept { return y; }

        constexpr float& getz() noexcept { return z; }

        // for const point32cpp objects
        constexpr float getx() const noexcept { return x; }

        constexpr float gety() const noexcept { return y; }

        constexpr float getz() const noexcept { return z; }
};

// we could also use a class hierarchy
class CPoint {
    protected:
        float _x;

    public:
        constexpr CPoint() noexcept : _x {} { }

        constexpr explicit CPoint(const float& x) noexcept : _x { x } { }

        constexpr float& getx() noexcept { return _x; }

        constexpr float getx() const noexcept { return _x; }
};

class CPoint2D : public CPoint {
    protected:
        float _y;

    public:
        constexpr CPoint2D() noexcept : CPoint {}, _y {} { }

        constexpr explicit CPoint2D(const float& x) noexcept : CPoint { x }, _y { x } { }

        constexpr CPoint2D(const float& x, const float& y) noexcept : CPoint { x }, _y { y } { }

        constexpr float& gety() noexcept { return _y; }

        constexpr float gety() const noexcept { return _y; }
};

class CPoint3D : public CPoint2D {
    protected:
        float _z;

    public:
        constexpr CPoint3D() noexcept : CPoint2D {}, _z {} { }

        constexpr explicit CPoint3D(const float& x) noexcept : CPoint2D { x }, _z { x } { }

        constexpr CPoint3D(const float& x, const float& y, const float& z) noexcept : CPoint2D { x, y }, _z { z } { }

        constexpr float& getz() noexcept { return _z; }

        constexpr float getz() const noexcept { return _z; }
};

// point3d could also be a class template
template<typename T, typename = std::enable_if<std::is_scalar<T>::value, T>::type> class TCPointD {
        using value_type      = T;
        using reference       = T&;
        using const_reference = const T&;

    private:
        value_type _x, _y, _z;

    public:
        constexpr TCPointD() noexcept : _x {}, _y {}, _z {} { }

        constexpr explicit TCPointD(const_reference v) noexcept : _x { v }, _y { v }, _z { v } { }

        constexpr explicit TCPointD(const_reference x, const_reference y, const_reference z) noexcept : _x { x }, _y { y }, _z { z } { }

        ~TCPointD() = default;

        [[nodiscard]] constexpr reference getx() noexcept { return _x; }

        [[nodiscard]] constexpr const_reference getx() const noexcept { return _x; }

        [[nodiscard]] constexpr reference gety() noexcept { return _y; }

        [[nodiscard]] constexpr const_reference gety() const noexcept { return _y; }

        [[nodiscard]] constexpr reference getz() noexcept { return _z; }

        [[nodiscard]] constexpr const_reference getz() const noexcept { return _z; }
};

// or it could use a buffer store the 3 values
template<typename T, typename = std::enable_if<std::is_scalar<T>::value, T>::type> requires requires {
    sizeof(T);
    std::is_scalar<typename std::remove_cv<T>::type>::value;
} class TCCPoint3D {
        using value_type      = T;
        using reference       = T&;
        using const_reference = const T&;

    private:
        value_type _buffer[3];

    public:
        constexpr TCCPoint3D() noexcept : _buffer {} { }

        constexpr explicit TCCPoint3D(const_reference v) noexcept : _buffer { v, v, v } { }

        constexpr explicit TCCPoint3D(const_reference x, const_reference y, const_reference z) noexcept : _buffer { x, y, z } { }

        ~TCCPoint3D() = default;

        [[nodiscard]] constexpr reference getx() noexcept { return _buffer[0]; }

        [[nodiscard]] constexpr const_reference getx() const noexcept { return _buffer[0]; }

        [[nodiscard]] constexpr reference gety() noexcept { return _buffer[1]; }

        [[nodiscard]] constexpr const_reference gety() const noexcept { return _buffer[1]; }

        [[nodiscard]] constexpr reference getz() noexcept { return _buffer[2]; }

        [[nodiscard]] constexpr const_reference getz() const noexcept { return _buffer[2]; }
};

int wmain() {
    const auto&& xref { ::NewPoint3D().x };
    point3dcpp   p3 { 4, 8, 16 };
    p3.getx() = std::numbers::pi_v<float>;

    auto& p3z { p3.getz() };

    constexpr point3dcpp p3c { 1, 4, 9 };
    p3c.getx() = std::numbers::pi_v<float>;

    CPoint3D CP3 { 1, 2, 3 };

    CP3.getx();

    constexpr CPoint3D _CP3 { 1, 2, 3 };
    _CP3.getx();

    constexpr TCPointD TCP3 { 3.875, 0.34121, 7.87425823 };
    TCP3.gety();

    constexpr TCCPoint3D<unsigned> TCCP3 { 5, 6, 7 };
    TCCP3.getz() = 987;

    return EXIT_SUCCESS;
}
