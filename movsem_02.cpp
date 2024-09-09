#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__ 0
#include <algorithm>
#include <sstring>

static inline std::string sentence_case(_In_ const std::string& asciistr) noexcept {
    std::string copy(asciistr);
    for (unsigned i = 0; i < copy.length(); ++i)
        // when the current char is ' ' and the next char is an ascii lowercase letter
        if (copy.at(i) == ' ' && copy.at(i + 1) >= 97 && copy.at(i + 1) <= 122) copy.at(i + 1) -= 32; // make it upper case
    return copy;
}

#if (__cplusplus >= 201103L)

static inline std::string sentence_case(_In_ std::string&& asciistr) noexcept {
    std::string copy(std::move(asciistr));
    for (unsigned i = 0; i < copy.length(); ++i)
        // when the current char is ' ' and the next char is an ascii lowercase letter make it upper case
        if (copy.at(i) == ' ' && copy.at(i + 1) >= 97 && copy.at(i + 1) <= 122) copy.at(i + 1) -= 32;
    return copy;
}

#endif

static inline ::sstring skyfall() noexcept {
    // NOLINTNEXTLINE(modernize-return-braced-init-list)
    return ::sstring("Skyfall is where we start, a thousand miles and poles apart, where worlds collide and days are dark!");
}

static __declspec(noinline) ::sstring no_time_to_die() noexcept { return "That I've fallen for a lie.... you were never on my side...."; }

static const size_t MiB = 1024 * 1024;

int main() {               // NOLINT(bugprone-exception-escape)
    const ::sstring empty; // default construction
    ::sstring       crab_style = ::sstring::with_capacity(7 * MiB);

    ::sstring jbond("I've drowned and dreamt this moment.... so overdue I owe them................");
    std::cout << jbond << '\n';

    ::sstring adele;     // default construction
    adele = ::skyfall(); // copy assignment in C++03, move assignment in C++11 and later

    std::cout << adele << '\n';

    const ::sstring aaaaa('A', 50);
    std::cout << aaaaa << '\n';

    const ::sstring concatenated = skyfall() + " " + jbond; // boy look at that :)
    std::cout << concatenated << '\n';

    const ::sstring no_time_to_die = ::no_time_to_die(); // move construction in C++11 and later, copy construction in in C++03 and before
    ::puts(no_time_to_die.c_str());

    jbond += " swept away I'm stooooolennnnnn..... let the sky fall.... when it crumbles we'll stand tall and face it all together...!";
    ::puts(jbond.c_str());

    for (const auto& c : jbond) {
        ::putchar(c);
        ::putchar('\n');
    }

    std::transform(jbond.begin(), jbond.end(), jbond.begin(), ::toupper);
    ::puts(jbond.c_str());

    const std::string james(jbond.cbegin(), jbond.cend()); // will work with -std=c++98 too :)
    std::cout << james << '\n';

    const std::string me    = "Anoban";
    ::sstring         metoo = me;
    std::cout << me << " " << metoo << '\n';

    std::string cohen = "A million candles burning for a help that never came!";
    ::puts(cohen.c_str());

    ::sstring want_it_darker = cohen; // copy construction
    ::puts(want_it_darker.c_str());

    want_it_darker = me; // copy assignment
    ::puts(want_it_darker.c_str());

    // invokes the conversion function to create a std::string from ::string, then the rvalue overload gets called in C++11 and later
    // const lvalue overload gets called in C++03 and before
    const std::string sentence = ::sentence_case(::skyfall());
    ::puts(sentence.c_str());

    return EXIT_SUCCESS;
}
