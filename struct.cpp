#include <cfloat>
#include <cstdint>
#include <utility>

constexpr size_t NELEMENTS { 256 };

struct container {
        double  dvalue;
        int64_t ivalue;
        float   array[NELEMENTS];
};

static constexpr container func() noexcept {
    float array[NELEMENTS];
    for (size_t i = 0; i < NELEMENTS; i++) array[i] = static_cast<float>(i);
    return container { .dvalue = DBL_MAX, .ivalue = INT64_MAX, .array = array }; // leveraging designated initializer lists
}

int main() { }
