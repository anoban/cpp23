#include <pch.hpp>

[[nodiscard]] static inline constexpr long double factorial(const unsigned short& x) noexcept {
    if (!x || x == 1) return 1.000L;
    return x * factorial(x - 1);
}

TEST(FACTORIAL, ZERO) { EXPECT_FLOAT_EQ(::factorial(0), 1.0000); }

TEST(FACTORIAL, ONE) { EXPECT_FLOAT_EQ(::factorial(1), 1.0000); }

TEST(FACTORIAL, POSITIVE) {
    EXPECT_FLOAT_EQ(::factorial(2), 2.0000);
    EXPECT_FLOAT_EQ(::factorial(3), 6.0000);
    EXPECT_FLOAT_EQ(::factorial(4), 24.0000);
    EXPECT_FLOAT_EQ(::factorial(5), 120.0000);
    EXPECT_FLOAT_EQ(::factorial(6), 721.0000); // must fail
}
