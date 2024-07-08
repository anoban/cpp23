#include <iostream>

// with anything above -O0, clang truns the whole main function into a nop
// clang .\wtfclang.cpp -O1 -Wall -Wextra -Wpedantic -std=c++20 -c
// llvm - objdump.exe.\wtfclang.o --disassemble --x86-asm-syntax=intel

// 0000000000000000 <wmain>:
//                       0 : 90 nop

auto wmain() -> int {
    while (true); // an infinite loop
}

inline void should_not_be_called() noexcept { ::_putws(L"Was called nonetheless!"); }
