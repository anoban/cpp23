#include <string>

static inline const wchar_t* greetme() {
    const std::wstring greeting { L"Hi there! how's the day been?" };
    return greeting.c_str();
}
