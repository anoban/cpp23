#include <array>
#include <cfloat>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <numeric>

constexpr size_t NELEMENTS { 256 };

struct container {
        double  dvalue;
        int64_t ivalue;
        float   array[NELEMENTS]; // NOLINT
};

struct container2 {
        double                       dvalue;
        int64_t                      ivalue;
        std::array<float, NELEMENTS> array;
};

#ifdef __TRYME__

static constexpr container func() noexcept {
    float array[NELEMENTS]; // NOLINT
    std::iota(array, array + __crt_countof(array), 0.000F);
    return container { .dvalue = DBL_MAX, .ivalue = INT64_MAX, .array = array }; // leveraging designated initializer lists
}

#endif // !__TRYME__

static constexpr container2 func2() noexcept {
    std::array<float, NELEMENTS> array {};
    std::iota(array.begin(), array.end(), 1.000F);
    return container2 { .dvalue = DBL_MAX, .ivalue = INT64_MAX, .array = array }; // leveraging designated initializer lists
    // okay because std::array has a defined copy ctor
}

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)

namespace anoban {
    template<typename scalar_t, size_t _size> requires std::integral<scalar_t> || std::floating_point<scalar_t> class array {
            using iterator       = scalar_t*;
            using const_iterator = const scalar_t*;
            using value_type     = scalar_t;

        private:
            size_t     size;
            value_type elements[_size]; // NOLINT

        public:
            // default ctor
            constexpr array() noexcept : size { _size }, elements { 0 } { ::_putws(L"anoban::array::array() noexcept"); }

            // copy ctor
            constexpr array(const array& other) noexcept : size { other.size } {
                ::_putws(L"anoban::array::array(const array& other) noexcept");
                ::memcpy_s(elements, size * sizeof(elements[0]), other.elements, other.size * sizeof(other.elements[0]));
            }

            // move ctor
            constexpr array(array&& other) = delete;

            // copy assignment
            constexpr array& operator=(const array& other) noexcept {
                ::_putws(L"anoban::array::operator=(const array& other) noexcept");
                if (this == &other) return *this;
                size = other.size;
                ::memcpy_s(elements, size * sizeof(elements[0]), other.elements, other.size, other.size * sizeof(other.elements[0]));
                return *this;
            }

            // move assignment
            constexpr array operator=(array&& other) = delete;

            // dtor
            constexpr ~array() noexcept {
                ::_putws(L"anoban::array::~array()");
                size = 0;
                ::memset(elements, 0U, sizeof(elements));
            }

            constexpr iterator       begin() noexcept { return elements; }

            constexpr iterator       end() noexcept { return elements + __crt_countof(elements) + 1; }

            constexpr const_iterator begin() const noexcept { return elements; } // for const objects

            constexpr const_iterator end() const noexcept { return elements + __crt_countof(elements) + 1; } // for const objects

            constexpr const_iterator cbegin() noexcept { return elements; }

            constexpr const_iterator cend() noexcept { return elements + __crt_countof(elements) + 1; }
    };

} // namespace anoban

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

struct container3 {
        double                          dvalue;
        int64_t                         ivalue;
        anoban::array<float, NELEMENTS> array;
};

static constexpr container3 func3() noexcept {
    anoban::array<float, NELEMENTS> array {};
    std::iota(array.begin(), array.end(), 1.000F);
    return { .dvalue { DBL_MAX }, .ivalue { INT64_MAX }, .array { array } }; // leveraging designated initializer lists
    // okay because anoban::array too has a defined copy ctor
}

int main() {
    auto y { func3() };
    return EXIT_SUCCESS;
}
