#define _AMD64_

#include <cstdio>
#include <cstdlib>
#include <numbers>

#include <processthreadsapi.h>
#pragma comment(lib, "ntdll.lib")

constexpr auto EXIT_CODE { 0b1010111 };
using NTSTATUS = signed int;

extern "C" {
    static void __cdecl SayGoodByeEx() noexcept { ::_putws(L"@exit :: Goodbye!"); }
    static void __cdecl SayGoodByeQEx() noexcept { ::_putws(L"@quick_exit :: Goodbye!"); }
    NTSTATUS NTAPI NtTerminateProcess(_In_opt_ HANDLE ProcessHandle, _In_ NTSTATUS ExitStatus);
}

class Object {
    public:
        Object() noexcept : value { std::numbers::egamma_v<float> } { ::_putws(L"ctor was called!"); }

        Object(const Object& other) noexcept : value { other.value } { ::_putws(L"copy ctor was called!"); }

        ~Object() noexcept {
            value = 0.0000F;
            ::_putws(L"dtor was called!");
        }

    private:
        float value {};
};

int main() {
    ::atexit(::SayGoodByeEx);
    ::at_quick_exit(::SayGoodByeQEx);

    auto object { Object {} };

    // return EXIT_SUCCESS;
    // ::exit(EXIT_CODE); // calling exit() prevents destructors being called in C++
    // ::quick_exit(EXIT_CODE); // calling quick_exit too prevents destructors being called in C++
    // ::abort(); // the worst possible option

    // ::ExitProcess(EXIT_CODE); // no dtor calls or calls to handlers registered with exit handlers
    const HANDLE64 hSelf = ::GetCurrentProcess();
    ::NtTerminateProcess(hSelf, EXIT_CODE); // no dtor calls or calls to handlers registered with exit handlers
}
