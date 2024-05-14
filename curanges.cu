#include <iostream>
#include <ranges>

static constexpr size_t ulimit { 100 };

auto main() -> int {
    for (const auto& i : std::ranges::views::iota(0LLU, ulimit)) ::wprintf_s(L"%zu\n", i);

    return EXIT_SUCCESS;
}
