#include <pch.hpp>

template<class... TList> [[nodiscard]] static constexpr long double sum(const TList&... args) noexcept { return (... + args); }

TEST(ALL_ZEROES, SUM) { EXPECT_EQ(::sum(0, 0.0, 0.0000L, 0.000F, 0L, 0U, 0LL, 0Ui16), 0); }

TEST(NON_ZEROES, SUM) {
    EXPECT_EQ(::sum(10, 10.0, 10.0000L, 10.000F, 10L, 10U, 10LL, 10Ui16), 80);
    EXPECT_EQ(::sum(10, 10.0, 11.0000L, 10.000F, 10L, 10U, 10LL, 10Ui16), 84);
}

TEST(DUMMY_TEST_SUITE, TEST_FALSE) {
    EXPECT_FALSE(nullptr);
    EXPECT_FALSE(NULL);
    EXPECT_FALSE(0);
}

TEST(DUMMY_TEST_SUITE, TEST_TRUE) {
    EXPECT_TRUE(1);
    EXPECT_TRUE(true);
    EXPECT_TRUE(reinterpret_cast<uintptr_t*>(0x76415AB543));
}
