#include <iostream>

template<typename _TyChar> concept is_isotreambuffer_compatible = std::is_same_v<_TyChar, char> || std::is_same_v<_TyChar, wchar_t>;

namespace nstd {

    template<typename _Ty> requires ::is_isotreambuffer_compatible<_Ty>
    [[nodiscard]] static constexpr std::basic_ostream<_Ty>& endl(std::basic_ostream<_Ty>& ostream) noexcept(noexcept(ostream << _Ty())) {
        ostream << _Ty('\n');
        return ostream;
    }

    template<typename _Ty> requires ::is_isotreambuffer_compatible<_Ty>
    [[nodiscard]] static constexpr std::basic_ostream<_Ty>& comma(std::basic_ostream<_Ty>& ostream) noexcept(noexcept(ostream << _Ty())) {
        ostream << _Ty(',');
        return ostream;
    }

    template<typename _Ty> requires ::is_isotreambuffer_compatible<_Ty>
    [[nodiscard]] static constexpr std::basic_ostream<_Ty>& wspace(std::basic_ostream<_Ty>& ostream) noexcept(noexcept(ostream << _Ty())) {
        ostream << _Ty(' ');
        return ostream;
    }

} // namespace nstd

template<typename _TyChar> requires ::is_isotreambuffer_compatible<_TyChar>
using ostream_manipulator_type = std::basic_ostream<_TyChar> & (*) (std::basic_ostream<_TyChar>&);

static_assert(std::is_same_v<decltype(&nstd::endl<wchar_t>), ostream_manipulator_type<wchar_t>>);

int wmain() {
    std::wcout << L"Hello there!" << nstd::endl;
    std::wcout << L"Did that print a newline??\n";

    for (const auto& e : L"It was an ordinary day....") std::wcout << e << nstd::comma << nstd::wspace;
    std::wcout << nstd::endl;

    return EXIT_SUCCESS;
}
