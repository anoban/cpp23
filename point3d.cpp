#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <ranges>
#include <type_traits>

// a POD C style struct (aggregate type)
struct point3d {
        float x, y, z; // no inclass initializers
};

#define printpoint3d(p3d) (printf("x:: %.4f, y:: %.4f, z:: %.4f\n", (p3d).x, (p3d).y, (p3d).z))

static_assert(sizeof(point3d) == 12);

// in C, we'd need a helper to print point32 to the console or the user code will have to unpack the struct and print it using a regular *printf* family function
static void PrintPoint3D(const point3d& point) noexcept { printf("Point3D :: (%.4f, %.4f, %.4f)\n", point.x, point.y, point.z); }

static constexpr point3d NewPoint3D() noexcept { return point3d { 0, 0, 0 }; }

// in C++, point3d is likely to be implemented as a class type (non-aggregate)
class point3dcpp final {
    private:
        float x, y, z;

    public:
        constexpr point3dcpp() noexcept : x {}, y {}, z {} { }

        constexpr explicit point3dcpp(const float& _val) noexcept : x { _val }, y { _val }, z { _val } { }

        constexpr point3dcpp(const float& _x, const float& _y, const float& _z) noexcept : x { _x }, y { _y }, z { _z } { }

        ~point3dcpp() = default;

        constexpr float get_x() const noexcept { return x; }

        constexpr float get_y() const noexcept { return y; }

        constexpr float get_z() const noexcept { return z; }
};

static_assert(sizeof(point3dcpp) == 12);

// we could also use a class hierarchy
class CPoint {
    protected:
        float _x;

    public:
        constexpr CPoint() noexcept : _x {} { }

        constexpr explicit CPoint(const float& x) noexcept : _x { x } { }

        constexpr float get_x() const noexcept { return _x; }
};

static_assert(sizeof(CPoint) == 4);

class CPoint2D : public CPoint {
    protected:
        float _y;

    public:
        constexpr CPoint2D() noexcept : CPoint {}, _y {} { }

        constexpr explicit CPoint2D(const float& x) noexcept : CPoint { x }, _y { x } { }

        constexpr CPoint2D(const float& x, const float& y) noexcept : CPoint { x }, _y { y } { }

        constexpr float get_y() const noexcept { return _y; }
};

static_assert(sizeof(CPoint2D) == 8);

class CPoint3D : public CPoint2D {
    protected:
        float _z;

    public:
        constexpr CPoint3D() noexcept : CPoint2D {}, _z {} { }

        constexpr explicit CPoint3D(const float& x) noexcept : CPoint2D { x }, _z { x } { }

        constexpr CPoint3D(const float& x, const float& y, const float& z) noexcept : CPoint2D { x, y }, _z { z } { }

        constexpr float get_z() const noexcept { return _z; }
};

static_assert(sizeof(CPoint3D) == 12);

// point3d could also be a class template
template<typename T, typename = std::enable_if<std::is_scalar<T>::value, T>::type> class TCPointD final {
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

        [[nodiscard]] constexpr value_type get_x() const noexcept { return _x; }

        [[nodiscard]] constexpr value_type get_y() const noexcept { return _y; }

        [[nodiscard]] constexpr value_type get_z() const noexcept { return _z; }
};

// or it could use a buffer store the 3 values
template<typename T> requires std::is_arithmetic_v<T> class TCCPoint3D final {
    public:
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

        [[nodiscard]] constexpr value_type get_x() const noexcept { return _buffer[0]; }

        [[nodiscard]] constexpr value_type get_y() const noexcept { return _buffer[1]; }

        [[nodiscard]] constexpr value_type get_z() const noexcept { return _buffer[2]; }

        template<typename char_t>
        friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostr, const TCCPoint3D& object) {
            for (const auto& e : object._buffer) ostr << e << char_t(',') << char_t(' ');
            ostr << char_t('\n');
            return ostr;
        }
};

int wmain() {
    const auto&& xref { ::NewPoint3D().x };
    point3dcpp   p3 { 4, 8, 16 };
    p3.get_x();

    auto&& p3z { p3.get_z() };

    constexpr point3dcpp p3c { 1, 4, 9 };
    p3c.get_x();

    CPoint3D CP3 { 1, 2, 3 };

    CP3.get_x();

    constexpr CPoint3D _CP3 { 1, 2, 3 };
    _CP3.get_x();

    constexpr TCPointD TCP3 { 3.875, 0.34121, 7.87425823 };
    TCP3.get_y();

    constexpr TCCPoint3D<unsigned> TCCP3 { 5, 6, 7 };
    TCCP3.get_z();

    printpoint3d(NewPoint3D());

    const auto [x, y, z] { NewPoint3D() }; // say hello to structed bindings

    auto p3d = NewPoint3D();
    PrintPoint3D(p3d);
    auto& [px, py, pz] = p3d;
    pz = py = px += 24.00;

    PrintPoint3D(p3d);
    return EXIT_SUCCESS;
}
