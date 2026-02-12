#include <array>
#include <numbers>
#include <ranges>
#include <string>
#include <vector>

// structured bindings

struct object {
        std::wstring         name;
        std::array<float, 3> values;
};

[[nodiscard]] static constexpr object __cdecl func(
) noexcept(std::is_nothrow_constructible<std::wstring>::value && std::is_nothrow_constructible<std::array<float, 3>>::value) {
    return {
        L"Anoban", { std::numbers::pi_v<float>, std::numbers::e_v<float>, std::numbers::sqrt2_v<float> }
    };
}

auto wmain() -> int {
    auto [name, arr] { ::func() };
    ::_putws(name.c_str());

    name = L"Julia";
    ::_putws(name.c_str());

    for (const auto& e : arr) printf("%.5f\n", e);

    auto& [pi, e, sqrt2]  = arr;
    pi                   *= 2;
    e                    *= 2;
    sqrt2                *= 2;

    for (const auto& e : arr) printf("%.5f\n", e);

    std::vector<object> collection(10);
    for (float x { 1.000 }; const auto& _ : std::ranges::views::iota(1, 10)) x *= _;

    return EXIT_SUCCESS;
}
