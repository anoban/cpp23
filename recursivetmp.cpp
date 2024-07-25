enum class month { January, February, March, April, May, June, July, August, September, October, November, December };

constexpr month& operator++(month& now) noexcept {
    if (now == month::December)
        now = month::January;
    else
        now = static_cast<month>(static_cast<int>(now) + 1);
    return now;
}

constexpr month operator++(month& now, int) noexcept {
    const auto temp { now };
    if (now == month::December)
        now = month::January;
    else
        now = static_cast<month>(static_cast<int>(now) + 1);
    return temp;
}

template<size_t n> struct factorial {
        static constexpr size_t value { n * factorial<n - 1>::value };
};

template<> struct factorial<0> {
        static constexpr size_t value { 1 };
};

static_assert(factorial<5>::value == 120);
static_assert(factorial<6>::value == 720);
static_assert(factorial<0>::value == 1);
static_assert(factorial<1>::value == 1);

template<typename... TList> [[nodiscard]] consteval double sum(const TList&... args) noexcept { return (args + ... + 0); }

template<typename... TList> [[nodiscard]] consteval double mul(const TList&... args) noexcept { return (... * args); }

static_assert(::sum(1, 2, 3, 4, 5, 6) == 21);
static_assert(::mul(1, 2, 3, 4, 5, 6) == 720);
static_assert(::mul(0, 1, 2, 3, 4, 5, 6) == 0);

static auto july { month::July };
