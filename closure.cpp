// clang .\closure.cpp -Wall -Wextra -std=c++20 -O3 -pedantic

#include <algorithm>
#include <cstdio>
#include <string>

// when a lambda is defined, the compiler creates a closure object
const auto printer = [](const std::wstring& wstring) noexcept -> void { ::_putws(wstring.c_str()); };

// compiler generated closure for the above lambda may look something like
struct __cpp20$$msvc15042024ucrt254_ThU202002LTtZ_printer$12__u35TiX_ {
        // since our lambda did not capture anything, we do not need any data members inside the functor
        void operator()(const std::wstring& wstring) const noexcept { ::_putws(wstring.c_str()); }
};

int main() {
    const std::wstring name { L"Anoban" };

    printer(name);
    __cpp20$$msvc15042024ucrt254_ThU202002LTtZ_printer$12__u35TiX_()(name); // ctor().operator()(name)

    constexpr auto closure { __cpp20$$msvc15042024ucrt254_ThU202002LTtZ_printer$12__u35TiX_ {} };
    closure(name);

    return EXIT_SUCCESS;
}
