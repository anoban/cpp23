#include <type_traits>
#include <cstring>
#include <algorithm>
#include <string>

static inline void capitalize(_Inout_ std::string& string) noexcept {	// use of a mutable reference as an in-out function argument
	std::transform(string.begin(), string.end(), string.begin(), ::toupper);
}

template<typename _TyChar, typename = typename std::enable_if<std::is_same<_TyChar, char>::value || std::is_same<_TyChar, wchar_t>::value, _TyChar>::type>
static inline void print(_In_ const std::basic_string<_TyChar>& string) noexcept {
	::wprintf_s(std::is_same<_TyChar, char>::value ? L"%S\n" : L"%s\n", string.c_str());
}

auto main() -> int {
	std::string name {"Anoban"};
	::capitalize(name);
	::print(name);
	return EXIT_SUCCESS;
}