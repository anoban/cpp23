// lambda captures

#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>

constexpr auto stateless = [](const int& _x) constexpr noexcept -> decltype(_x) { return _x; };

constexpr void func() noexcept {
    const std::wstring monkey { L"Chulmondely" };
    const auto         pi       = std::numbers::pi_v<float>;
    auto               stateful = [&pi, &monkey](const double& _r) constexpr noexcept -> double {
        ::_putws(monkey.c_str());
        return pi * _r * _r;
    };
    constexpr auto stateful_size = sizeof(stateful);
    stateful(12.0);
}

int wmain() {
    constexpr auto stateless_size = sizeof(stateless);
    func();
    static const std::wstring name { L"DELL" };

    return EXIT_SUCCESS;
}
