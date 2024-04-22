#include <cstdio>
#include <string>

int main() {
    const char* const name { "Anoban" }; // string literal in .rdata, pointer non-modifiable

    auto* const home { R"(C\Users\Natalie\Documents\Misc)" }; // pointer non-modifiable

    puts(name);
    puts(home);

    auto& refto_name { name }; // reference to a constant pointer to constant char
    refto_name = "Jhonathan";  // Error: expression must be a modifiable lvalue
    refto_name = home;         // Error: expression must be a modifiable lvalue

    auto laptop { L"MSI" };     // wide string literal in .rdata, pointer non const qualified
    auto mobile { L"SAMSUNG" }; // wide string literal in .rdata, pointer non const qualified

    auto& refto_laptop { laptop }; // reference to a modifiable pointer to a constant wchar_t
    refto_laptop =
        L"HP"; // okay because this doesn't modify the string literal, this created a new string literal in .rdata and assigned its
    // address to laptop.
    // again modifying a reference doesn't do anythin to the reference itself, it realizes the effects on the referred object

    // if laptop and mobile were const qualified pointers, we won't be able to do this because refto_laptop = L"HP" will be a request to
    // modify a constant pointer

    return EXIT_SUCCESS;
}
