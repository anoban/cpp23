#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 1
#include <sstring>
#include <algorithm>

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

    std::cout << "__cplusplus is " << __cplusplus << '\n';
    ::getchar();

    return EXIT_SUCCESS;
}