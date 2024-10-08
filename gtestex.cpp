#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

template<typename _Ty> static constexpr std::add_rvalue_reference<_Ty>::type declval() noexcept;

namespace nstd {
    template<typename iterator, typename accumulator, typename = std::enable_if<decltype(*iterator {}), decltype(iterator { } ++)>::type>
    static constexpr inline accumulator reduce(
        _In_ const iterator& _start, _In_ const iterator& _end, _In_ const accumulator& _accum
    ) noexcept {
        accumulator aggregate(_accum);
        for (iterator beg = _start, end = _end; beg != end; ++beg) aggregate += *beg;
        return aggregate;
    }
} // namespace nstd

auto main() -> int {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
