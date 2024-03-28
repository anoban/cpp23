// print out the version of commctrl DLL available on this machine

// clang-format off
#define _AMD64_
#include <windef.h>
#include <WinBase.h>
#include <Shlwapi.h>
// clang-format on
#include <iostream>

auto main(void) -> int {
    constexpr auto  ComCtl32 { L"C:/Windows/System32/ComCtl32.dll" };
    const HINSTANCE hDllInstance { LoadLibraryW(ComCtl32) };

    if (hDllInstance) {
        const DLLGETVERSIONPROC DllGetVersion { reinterpret_cast<DLLGETVERSIONPROC>(GetProcAddress(hDllInstance, "DllGetVersion")) };
        if (DllGetVersion) {
            DLLVERSIONINFO                 DllVersionInfo { .cbSize = sizeof(DLLVERSIONINFO2) };
            [[maybe_unused]] const HRESULT hRes { (*DllGetVersion)(&DllVersionInfo) };
            std::wcout << DllVersionInfo.dwMajorVersion << L'.' << DllVersionInfo.dwMinorVersion << L' ' << DllVersionInfo.dwBuildNumber
                       << L' ' << DllVersionInfo.dwPlatformID << L'\n';
        };
        FreeLibrary(hDllInstance);
        return 0;
    }
    return -10;
}
