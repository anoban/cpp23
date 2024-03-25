#include <iostream>
#include <string>

class greeter {
        std::wstring greeting;

    public:
        greeter() = delete;
        inline greeter(std::wstring msg) noexcept : greeting { std::move(msg) } { }
        ~greeter() = default;
        inline std::wstring          operator()() const noexcept { return greeting; }
        inline friend std::wostream& operator<<(std::wostream& ostr, const greeter& obj) noexcept {
            ostr << obj.greeting << L'\n';
            return ostr;
        }
};

auto main(void) -> int {
    const auto functor { greeter { L"Hi there Ano!" } };
    std::wcout << functor();
    return EXIT_SUCCESS;
}
