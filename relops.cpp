#include <concepts>
#include <cstdlib>
#include <iostream>
#include <type_traits>

// NOLINTBEGIN(modernize-use-constraints,readability-redundant-inline-specifier,cppcoreguidelines-missing-std-forward,bugprone-use-after-move)

namespace derived_comparisons { // expects that the passed types suppport == and < operators

    template<class _TyL, class _TyR> concept is_equality_comparable  = requires { std::declval<_TyL>() == std::declval<_TyR>(); };

    template<class _TyL, class _TyR> concept is_less_than_comparable = requires { std::declval<_TyL>() < std::declval<_TyR>(); };

    template<class _TyL, class _TyR> requires is_equality_comparable<_TyL, _TyR>
    static inline constexpr bool operator!=(_In_ _TyL&& left, _In_ _TyR&& right) noexcept {
        return !(std::forward<_TyL>(left) == std::forward<_TyR>(right));
    }

    template<class _TyL, class _TyR> requires(is_less_than_comparable<_TyL, _TyR> && is_equality_comparable<_TyL, _TyR>)
    static inline constexpr bool operator<=(_In_ _TyL&& left, _In_ _TyR&& right) noexcept {
        return std::forward<_TyL>(left) == std::forward<_TyR>(right) || std::forward<_TyL>(left) < std::forward<_TyR>(right);
    }

    template<class _TyL, class _TyR> requires(is_less_than_comparable<_TyL, _TyR> && is_equality_comparable<_TyL, _TyR>)
    static inline constexpr bool operator>(_In_ _TyL&& left, _In_ _TyR&& right) noexcept {
        return !(std::forward<_TyL>(left) == std::forward<_TyR>(right) || std::forward<_TyL>(left) < std::forward<_TyR>(right));
    }

    template<class _TyL, class _TyR> requires is_less_than_comparable<_TyL, _TyR>
    static inline constexpr bool operator>=(_In_ _TyL&& left, _In_ _TyR&& right) noexcept {
        return !(std::forward<_TyL>(left) < std::forward<_TyR>(right));
    }

} // namespace derived_comparisons

// NOLINTEND(modernize-use-constraints,readability-redundant-inline-specifier,cppcoreguidelines-missing-std-forward,bugprone-use-after-move)

class foo final { };

class bar final {
    public:
        // operators only usable with rvalues
        constexpr bool operator==(_In_ [[maybe_unused]] const foo& other) const&& noexcept { return false; }
        constexpr bool operator<(_In_ [[maybe_unused]] const foo& other) const&& noexcept { return false; }
};

static __declspec(noinline) constexpr bool func(_In_ const bar& b, _In_ const foo& f) noexcept { return b < f; }
static __declspec(noinline) constexpr bool func(_In_ bar&& b, _In_ foo&& f) noexcept { return b < f; }

struct bazz final {
        // operators usable with lvalues and rvalues
        constexpr bool operator==(_In_ [[maybe_unused]] const foo& other) const& noexcept { return true; }
        constexpr bool operator<(_In_ [[maybe_unused]] const foo& other) const& noexcept { return true; }
};

struct wow final {
        // operators only usable with lvalues
        constexpr bool operator==(_In_ [[maybe_unused]] const foo& other) const& noexcept { return true; }
        constexpr bool operator<(_In_ [[maybe_unused]] const foo& other) const& noexcept { return true; }
        // because we've explicitly deleted the rvalue overloads
        constexpr bool operator==(_In_ [[maybe_unused]] const foo& other) const&& noexcept = delete;
        constexpr bool operator<(_In_ [[maybe_unused]] const foo& other) const&& noexcept  = delete;
};

int wmain() {
    const auto something { bar {} };
    const bool invalid { something < foo {} }; // candidate function not viable: expects an rvalue for object argument
    const bool valid { bar {} < foo {} };

    std::wcout << std::boolalpha;

    {
        using namespace derived_comparisons; // NOLINT(google-build-using-namespace)

        const auto something { bar {} };
        const auto otherthing { foo {} };

        const bool invalid { something < foo {} }; // candidate function not viable: expects an rvalue for object argument
        // candidate template ignored: constraints not satisfied [with _TyL = const bar &, _TyR = const foo &]
        const bool invalid_too { something != otherthing };

        const bool valid { bar {} < foo {} };

        std::wcout << (bar {} != foo {}) << L'\n';
        std::wcout << (bar {} <= foo {}) << L'\n';
        std::wcout << (bar {} > foo {}) << L'\n';
        std::wcout << (bar {} >= foo {}) << L'\n';

        auto&& xvref_bar { bar {} };
        auto&& xvref_foo { foo {} };

        const bool okay { xvref_bar <= xvref_foo };  // xvalues - lvalues with a type signature of rvalue references
        const bool mhmmm { xvref_bar == xvref_foo }; // because xvalues are not rvalues
        ::func(xvref_bar, xvref_foo);
        ::func(bar {}, foo {});
    }

    {
        using namespace derived_comparisons;

        foo  royal {};
        bazz what {};

        const bool valid { what == royal };
        const bool valid_too { what < royal };

        const bool nope { bazz {} == royal };

        // in calss bazz, operators < and == are const methods taking an lvalue reference and rvalues can bind to const lvalue references
        // if we do not want this behaviour we need to explicitly delete those comparison operators taking rvalue references
        std::wcout << (bazz {} != foo {}) << L'\n';
        std::wcout << (bazz {} <= foo {}) << L'\n';
        std::wcout << (bazz {} > foo {}) << L'\n';
        std::wcout << (bazz {} >= foo {}) << L'\n';
    }

    {
        using namespace derived_comparisons;

        foo royal {};
        wow what {};

        const bool valid { what == royal };
        const bool valid_too { what < royal };

        const bool nope { wow {} == royal };

        std::wcout << (wow {} != foo {}) << L'\n';
        std::wcout << (wow {} <= foo {}) << L'\n';
        std::wcout << (wow {} > foo {}) << L'\n';
        std::wcout << (wow {} >= foo {}) << L'\n';

        std::wcout << (what != foo {}) << L'\n';
        std::wcout << (what <= foo {}) << L'\n';
        std::wcout << (what > foo {}) << L'\n';
        std::wcout << (what >= foo {}) << L'\n';
    }

    return EXIT_SUCCESS;
}
