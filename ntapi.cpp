#define _AMD64_
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN

// clang-format off
#include <windef.h>
typedef LONG NTSTATUS ;
#include <processthreadsapi.h>
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <type_traits>

#pragma comment(lib, "ntdll.lib") // NtTerminateProcess

constexpr auto EXIT_CODE { 0b1010111 };

extern "C" {
    // function to be called when exit() is invoked
    static void __cdecl SayGoodByeEx() noexcept { ::_putws(L"@exit :: Goodbye!"); }

    // function to be called when quick_exit is called
    static void __cdecl SayGoodByeQEx() noexcept { ::_putws(L"@quick_exit :: Goodbye!"); }

    // declaration for NtTerminateProcess, which is not avaliable in the SDK headers
    NTSTATUS NTAPI NtTerminateProcess(_In_opt_ HANDLE ProcessHandle, _In_ NTSTATUS ExitStatus);
}

// a dummy class to test what happens to destructors in case of an unconventional exit
template<typename T, typename = std::is_arithmetic<T>::type> class object {
    public:
        using value_type = std::remove_cv_t<T>;

    private:
        value_type _value;

    public:
        object() noexcept : _value() { }

        object(const object&)            = delete;

        object(const object&&)           = delete;

        object& operator=(const object&) = delete;

        object& operator=(object&&)      = delete;

        ~object() noexcept { _value = ::_putws(L"dtor was calleed!"); }
};

int wmain() {
    // registering the exit handlers with the runtime
    ::atexit(::SayGoodByeEx);
    ::at_quick_exit(::SayGoodByeQEx);

    const auto     obj { object<double> {} };

    // return EXIT_SUCCESS;

    // ::exit(EXIT_CODE); // calling exit() prevents destructors being called in C++

    // ::quick_exit(EXIT_CODE); // calling quick_exit too prevents destructors being called in C++

    // ::abort(); // the worst possible option

    // ::ExitProcess(EXIT_CODE); // no dtor calls or calls to the routines registered with exit handlers

    const HANDLE64 hSelf = ::GetCurrentProcess();
    // ::TerminateProcess(hSelf, EXIT_CODE);    // no dtor calls and no calls to exit handlers

    ::NtTerminateProcess(hSelf, EXIT_CODE); // no dtor calls or calls to routines registered with exit handlers
}
