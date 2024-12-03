// g++ closures.cpp -Wall -Wextra -Wpedantic -O3 -std=c++20

#include <iostream>
#include <ranges>
#include <string>

// a closure object with two captures :: name and ncalls
struct greeter {
        static const size_t charsname { 100 };

        greeter() = delete;

        explicit greeter(const wchar_t* const _name) throw() : name { 0 }, ncalls { 0 } { ::wcsncpy_s(name, _name, charsname); }

        const wchar_t* get_name() const throw() { return name; }

        // operator() captures the member variables name and ncalls for use!
        void           operator()() {
            std::wcout << name << L" said helloooo!\n";
            ncalls++;
        }

        constexpr size_t count() const throw() { return ncalls; }

    private:
        wchar_t name[charsname];
        size_t  ncalls;
};

int main() {
    auto santa { greeter { L"Santa" } };

    for (const auto& _ : std::ranges::views::iota(0, 100)) santa();
    std::wcout << L"Santa has been asked to greet " << santa.count() << L" times!";

    return EXIT_SUCCESS;
}
