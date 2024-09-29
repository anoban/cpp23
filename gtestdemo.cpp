#include <gtest/gtest.h>

template<class... TList> [[nodiscard]] static constexpr long double sum(const TList&... args) noexcept { return (... + args); }

TEST(SUM, ZEROES) {
    //
    EXPECT_DOUBLE_EQ(::sum(0.000, 0LL, 0U, 0.0000F, 0, 0Ui16), 0.00000);
}

auto main() -> int {
    //

    return EXIT_SUCCESS;
}
