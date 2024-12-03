#include <gtest/gtest.h>

// g++ -std=c++20 -Wall -Wextra -O3 -I./googletest/ -I./googletest/include/ ./googletest/src/gtest-all.cc gtestdemo.cpp  -static
// clang .\gtestdemo.cpp -Wall -Wextra -static -march=native -O3 -std=c++20 -I.\googletest\ -I.\googletest\include\ .\googletest\src\gtest-all.cc

template<class... TList> [[nodiscard]] static constexpr long double sum(const TList&... args) noexcept { return (... + args); }

TEST(SUM, ZEROES) {
    //
    EXPECT_DOUBLE_EQ(::sum(0.000, 0LL, 0U, 0.0000F, 0, '\0'), 0.00000);
}

class Fixture : public testing::Test {
    protected:
        float       _value_0;
        float       _value_1;

        // inline Fixture() noexcept : testing::Test(), _value_0(10.0000), _value_1(11.00000) { }

        inline void SetUp() noexcept override {
            _value_0 = 100.000;
            _value_1 = 110.000;
        }

        // inline ~Fixture() noexcept override = default;

        inline void TearDown() noexcept override { _value_0 = _value_1 = 0.00000; }

        inline void SayHi() noexcept { ::puts("Hi there!"); }
};

TEST_F(Fixture, IS_TEN_AND_ELEVEN) {
    EXPECT_FLOAT_EQ(_value_0, 100.0000);
    EXPECT_FLOAT_EQ(_value_1, 110.0000);
    this->SayHi();
    EXPECT_NE(_value_0, 10.0000);
    EXPECT_NE(_value_1, 11.0000);
}

// The fixture for testing class Foo.
class FooTest : public testing::Test {
    protected:
        // You can remove any or all of the following functions if their bodies would
        // be empty.

        FooTest() noexcept {
            // You can do set-up work for each test here.
        }

        ~FooTest() noexcept override {
            // You can do clean-up work that doesn't throw exceptions here.
        }

        // If the constructor and destructor are not enough for setting up
        // and cleaning up each test, you can define the following methods:

        void SetUp() noexcept override {
            // Code here will be called immediately after the constructor (right
            // before each test).
        }

        void TearDown() noexcept override {
            // Code here will be called immediately after each test (right
            // before the destructor).
        }

        // Class members declared here can be used by all tests in the test suite
        // for Foo.
};

auto main() -> int {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
