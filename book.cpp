#include <concepts>
#include <cstdio>

template<typename T> requires std::integral<T> class book {
    private:
        T nborrows {};

    public:
        book() = delete;

        explicit constexpr book(T n) noexcept : nborrows { n } { }

        constexpr book& operator++() noexcept {
            nborrows++;
            return *this;
        }

        constexpr book& operator--() noexcept {
            nborrows--;
            return *this;
        }
};

int main() {
    long x { 45 };
    auto gbs { book { 565LL } };

    // prefix ++ and -- operators return a reference
    ::wprintf_s(L"%ld\n", ++x);
    ::wprintf_s(L"%ld\n", x);
    --x = 0;
    ::wprintf_s(L"%ld\n", x);

    // postfix ++ and -- operators return a value
    x++ = 0; // Error: expression must be a modifiable lvalue

    return 0;
}
