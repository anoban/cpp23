#include <type_traits>

template<class T> struct regular {
        short _w;
        float _x;
        T     _y;
        char  _z;
};

#pragma pack(push, 1)
template<class T> struct packed {
        T     _y;
        short _w;
        float _x;
        char  _z;
};
#pragma pack(pop)

static_assert(sizeof(regular<double>) == 24);
static_assert(sizeof(packed<double>) == 15);

template<bool _packed, class T> requires std::is_arithmetic_v<T> static consteval T factory([[maybe_unused]] const T& dummy) noexcept {
    return static_cast<T>(_packed);
}

auto wmain() -> int {
    //
    constexpr auto zero { ::factory<false>(5.0907) };
    constexpr auto one { ::factory<true>(L'A') };
    return 0;
}
