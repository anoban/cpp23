#include <cstdalign>
#include <cstddef>
#include <cstdint>

template<unsigned eval> static unsigned evaluate() noexcept { return eval; }

struct align {
        bool      first;
        int16_t   second;
        double    third;
        char      fourth;
        long long fifth;
};

struct __declspec(align(128)) align_ {
        bool      first;
        int16_t   second;
        double    third;
        char      fourth;
        long long fifth;
        float     sixth;
};

// compile time constants can be...
int main() {
    evaluate<sizeof(double)>(); // okay
    evaluate<alignof(align::fifth)>();
    evaluate<alignof(align_)>();
    evaluate<offsetof(align_, fourth)>();
    evaluate<19>();

    return 0;
}
