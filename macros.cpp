#include <cstdio>
#include <cstdlib>

#define __helper__(expanded_macro) L##expanded_macro // token paste
#define to_wstring(macro)          __helper__(macro)

#define NAME                       "ANOBAN"

static constexpr auto Anoban { to_wstring(NAME) };
static constexpr auto Dell { to_wstring("DELL") };

#define FUNC_FULL_NAME to_wstring(__FUNCSIG__)

auto wmain() -> int {
    constexpr auto main { to_wstring(__FUNCSIG__) };
    ::_putws(main);
    ::_putws(Anoban);
    ::_putws(Dell);

    ::_putws(FUNC_FULL_NAME);

#ifdef __llvm__
    ::puts(__PRETTY_FUNCTION__);
#endif

    return EXIT_SUCCESS;
}
