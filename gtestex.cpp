#include <type_traits>

#include <gtest/gtest.h>

template<typename _Ty0, typename... _TyList> struct get_first final {
        using type = _Ty0;
};

template<typename _TyCandidate, typename... _TyList> struct is_in final {
        static constexpr bool value =
            std::is_same<_TyCandidate, typename ::get_first<_TyList...>::type>::value || is_in<_TyCandidate, _TyList...>::value;
};

static_assert(::is_in<float, char, wchar_t, const double, long&, volatile float&&>::value);

TEST(IS_IN, EMPTY_TYPE_LIST) {
    EXPECT_FALSE();
    //
}

auto main() -> int {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
