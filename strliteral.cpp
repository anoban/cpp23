#include <cstdio>

int main() {
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
    return 0;
}
