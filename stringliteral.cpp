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

    const auto*       stackoverflow { "What you did is modify i through the reference ri. Print i after, and you'll see this." };
    // stackoverflow is a pointer to constant string literal in the .rdata section
    const char* const cstackoverflow { "It seems as if I have actually succeeded in reassigning a reference. Is that true?" };
    // cstackoverflow is a constant pointer to constant string literal in the .rdata section

    const auto* What_you_did { stackoverflow }; // we capture the address where this string literal is stored
    stackoverflow = "Stack overflow is the most toxic place one could go to seek help with programming questions!";
    // what happened above is that we created a new constant string literal in .rdata and assigned it to the pointer stackoverflow
    // we did not overwrite "What you did is modify i through the reference ri. Print i after, and you'll see this." !!!!

    puts(What_you_did);
    puts(stackoverflow);

    // we cannot do the same with cstackoverflow, because here in addition to the string literal, the pointer itself is a constant
    cstackoverflow = "see, we cannot modify a constant pointer!";
    // Error: cannot assign to variable 'cstackoverflow' with const-qualified type 'const char *const'

    return EXIT_SUCCESS;
}
