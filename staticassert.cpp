#include <cassert>
#include <cstddef>
#include <cstdlib>

struct node_t {
        unsigned char byte;
        unsigned      frequency;
};

static_assert(!offsetof(node_t, byte));
static_assert(!false);
static_assert(!0);

auto wmain() -> int { return EXIT_SUCCESS; }
