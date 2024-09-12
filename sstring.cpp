#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 0

#include <iostream>
#include <ranges>
#include <sstring>

auto main() -> int {
    ::sstring  name { "Anoban" };
    const auto me { name };

    std::cout << std::boolalpha;
    std::cout << (name == "Anoban ") << '\n';
    std::cout << (name == me) << '\n';

    for (const auto& _ : std::ranges::views::iota(0, 100)) {
        name += " Anoban";
        std::cout << "length = " << name.length() << " capacity = " << name.capacity() << '\n';
    }
    std::cout << name << '\n';

    const ::sstring skyfall { "I've drowned and dreamt this moment! I'm swept awaaaaaay I'm stolennnnnnnnnn!" };
    std::cout << skyfall << '\n';

    ::sstring bane { "But no ordinary child, a child born in hell, forged from suffering and hardened by pain, not a man from privilege!" };

    return EXIT_SUCCESS;
}
