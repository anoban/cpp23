#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 1
#include <sstring>
#include <algorithm>

static inline char __stdcall sentence_case(const char* const str) throw(){
    if(*str == ' ' && *(str + 1) >= 97 && *(str + 1) <= 122) return (*str) - 32;
    return *str;
}

int main(){
    const ::sstring brennan = "I remember that time we were sippin' on wine";
    std::cout << brennan << '\n';

    ::sstring copy = brennan;
    std::cout << copy << '\n';    
    std::transform(copy.cbegin(), copy.cend(), copy.begin(), ::toupper);
    std::cout << copy << '\n';    

    ::sstring lithe = "heard you want a piece of the pie huh?";
    lithe += " all the smokin' made me tired! Bitch I see the legacy , It's qiet!\n";
    std::cout << lithe;    

    for(unsigned i = 0; i < lithe.length(); ++i) lithe.at(i) = ::sentence_case(lithe.c_str() + i);
    std::cout << lithe;    

    std::cout << "__cplusplus is " << __cplusplus << '\n';
    ::getchar();

    return EXIT_SUCCESS;
}