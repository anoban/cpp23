#include <cstdio>
#include <cstdlib>

// https://learn.microsoft.com/en-us/cpp/preprocessor/preprocessor-experimental-overview?view=msvc-170

#define __helper__(expanded_macro) L##expanded_macro // token paster
#define to_wstring(macro)          __helper__(macro) // user facing macro expander

#define NAME                       "ANOBAN"

static constexpr auto Anoban { to_wstring(NAME) };
static constexpr auto Dell { to_wstring("DELL") };

#if (defined __llvm__ && defined __clang__) || (defined __GNUC__ && defined __GNUG__)
    #define FUNC_FULL_NAME to_wstring(__PRETTY_FUNCTION__) // for LLVM and g++
#elif defined _MSC_VER && defined _MSC_FULL_VER
    #if !defined(_MSVC_TRADITIONAL) || (_MSVC_TRADITIONAL) // not defined or defined to be 0
        #error Use the new preprocessor!
    #endif
    #define FUNC_FULL_NAME to_wstring(__FUNCSIG__) // for MSVC
#endif

static constexpr decltype(auto) signature() noexcept { return FUNC_FULL_NAME; }

auto wmain() -> int {
    ::_putws(Anoban);
    ::_putws(Dell);

    ::_putws(FUNC_FULL_NAME);
    ::_putws(::signature());

    return EXIT_SUCCESS;
}
