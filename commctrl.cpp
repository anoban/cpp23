// print out the version of commctrl DLL available on this machine

// clang-format off
#include <cstdlib>
#include <iostream>
#include <Shlwapi.h>
// clang-format on

auto main() -> int {
    constexpr auto ComCtl32 { L"C:/Windows/System32/ComCtl32.dll" };
    HINSTANCE      hDllInstance { ::LoadLibraryW(ComCtl32) };

    if (hDllInstance) {
        DLLGETVERSIONPROC dllGetVersion { reinterpret_cast<DLLGETVERSIONPROC>(::GetProcAddress(hDllInstance, "DllGetVersion")) };
        if (dllGetVersion) {
            DLLVERSIONINFO                 dllVersionInfo { .cbSize = sizeof(DLLVERSIONINFO2) };
            [[maybe_unused]] const HRESULT hRes { (*dllGetVersion)(&dllVersionInfo) };
            std::wcout << dllVersionInfo.dwMajorVersion << L'.' << dllVersionInfo.dwMinorVersion << L' ' << dllVersionInfo.dwBuildNumber
                       << L' ' << dllVersionInfo.dwPlatformID << L'\n';
        };
        ::FreeLibrary(hDllInstance);
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}
